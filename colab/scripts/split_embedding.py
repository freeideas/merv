#!/usr/bin/env python
"""Split the fp16 token-embedding so no single WebGPU buffer exceeds the adapter's
`maxBufferSize`. Post-process step that runs AFTER convert_to_onnx.py.

Why this exists (browser-load ceiling #3 in the README):
  convert_to_onnx.py casts the token-embedding `[200064, 3072]` to fp16 to clear the
  ~4 GB WASM heap. But that leaves the embedding as ONE 1.23 GB initializer, i.e. one
  1.23 GB GPU buffer. WebGPU caps a single buffer at the adapter's `maxBufferSize` --
  commonly 1 GiB on integrated/software adapters -- so the model fails to *run* with:
    "Buffer size (1229193216) exceeds the max buffer size limit (1073741824)"
  You can't raise the limit from JS: ORT-web already calls requestDevice() with
  requiredLimits = the adapter's maximum. The only fix is a smaller buffer.

What it does (lossless -- identical fp16 values, just reorganized):
  - splits the embedding `[V, H]` into N column-slices `[V, H/N]` (N=4 -> ~307 MB ea)
  - replaces the single `Gather` with N `Gather`s (same indices) + a `Concat(axis=-1)`
  - re-packs all external tensors into fresh <2 GB shards + a new manifest

RAM-safe: streams tensors one at a time (never loads the whole model into memory).

Usage:
  python split_embedding.py SRC_ONNX_DIR OUT_ONNX_DIR [--n 4] [--max-shard 1900000000]
    SRC_ONNX_DIR : dir with model_q4.onnx + *.onnx_data_* + external_data_manifest.json
    OUT_ONNX_DIR : written fresh (model_q4.onnx + new shards + new manifest)
"""
import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper

EXTERNAL = TensorProto.EXTERNAL
FP16_BYTES = 2


def _ext(t):
    """external_data entries -> {location, offset, length} (ints where numeric)."""
    d = {e.key: e.value for e in t.external_data}
    return d["location"], int(d["offset"]), int(d["length"])


def find_embedding(g):
    """The embedding = the largest fp16 initializer used as a Gather's data input."""
    inits = {t.name: t for t in g.initializer}
    best = None  # (size, init, gather_node)
    for n in g.node:
        if n.op_type != "Gather" or not n.input:
            continue
        t = inits.get(n.input[0])
        if t is None or t.data_type != TensorProto.FLOAT16:
            continue
        size = int(np.prod(t.dims)) * FP16_BYTES
        if best is None or size > best[0]:
            best = (size, t, n)
    if best is None:
        raise SystemExit("no fp16 initializer feeding a Gather found -- nothing to split")
    return best[1], best[2]


def split(src_dir, out_dir, n_split, max_shard):
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(src_dir, "model_q4.onnx")
    m = onnx.load(graph_path, load_external_data=False)
    g = m.graph

    emb, gather = find_embedding(g)
    # only the Gather may consume the embedding (else the rewrite would drop a user)
    users = [nd.name for nd in g.node if emb.name in nd.input]
    assert users == [gather.name], f"embedding has unexpected users: {users}"
    gather_out = gather.output[0]                 # keep the same output name -> consumers unchanged
    axis = next((a.i for a in gather.attribute if a.name == "axis"), 0)
    assert axis == 0, f"expected Gather axis=0 (row lookup), got {axis}"

    vocab, hid = [int(d) for d in emb.dims]
    if hid % n_split:
        raise SystemExit(f"hidden dim {hid} not divisible by --n {n_split}")
    part = hid // n_split
    loc, off, length = _ext(emb)
    print(f"embedding {emb.name} [{vocab},{hid}] fp16 @ {loc} off={off} len={length}")

    # read embedding once, slice into N contiguous column blocks
    with open(os.path.join(src_dir, loc), "rb") as f:
        f.seek(off)
        arr = np.frombuffer(f.read(length), dtype=np.float16).reshape(vocab, hid)
    part_names = [f"{emb.name}.part{k}" for k in range(n_split)]
    part_bytes = {
        part_names[k]: np.ascontiguousarray(arr[:, k * part:(k + 1) * part]).tobytes()
        for k in range(n_split)
    }
    del arr
    print(f"split into {n_split} x [{vocab},{part}] = "
          f"{len(part_bytes[part_names[0]]) / 1e6:.0f} MB each")

    # graph rewrite: drop embedding init + Gather; add N part inits, N Gathers, 1 Concat
    g.initializer.remove(emb)
    for name in part_names:
        t = TensorProto()
        t.name = name
        t.data_type = TensorProto.FLOAT16
        t.dims.extend([vocab, part])
        t.data_location = EXTERNAL
        g.initializer.append(t)

    idx = next(i for i, nd in enumerate(g.node) if nd.name == gather.name)
    del g.node[idx]
    new_nodes, gather_outs = [], []
    for k, name in enumerate(part_names):
        o = f"{gather_out}_part{k}"
        gather_outs.append(o)
        new_nodes.append(helper.make_node(
            "Gather", [name, gather.input[1]], [o],
            name=f"{gather.name}_part{k}", axis=0))
    new_nodes.append(helper.make_node(
        "Concat", gather_outs, [gather_out], name=f"{gather.name}_concat", axis=-1))
    for j, nd in enumerate(new_nodes):
        g.node.insert(idx + j, nd)

    # re-pack every external initializer into fresh shards (stream one at a time)
    handles = {}

    def read_old(t):
        l, o, n = _ext(t)
        h = handles.setdefault(l, open(os.path.join(src_dir, l), "rb"))
        h.seek(o)
        return h.read(n)

    ext_inits = [t for t in g.initializer if t.data_location == EXTERNAL]
    print(f"re-packing {len(ext_inits)} external tensors ...")
    shard_names, shard_idx, shard_off = [], 0, 0

    def shard_name(i):
        return f"model_q4.onnx_data_{i}"

    def open_shard(i):
        shard_names.append(shard_name(i))
        return open(os.path.join(out_dir, shard_name(i)), "wb")

    sf = open_shard(0)
    for t in ext_inits:
        data = part_bytes.get(t.name) or read_old(t)
        if shard_off and shard_off + len(data) > max_shard:
            sf.close()
            shard_idx += 1
            shard_off = 0
            sf = open_shard(shard_idx)
        sf.write(data)
        del t.external_data[:]
        for key, val in (("location", shard_name(shard_idx)),
                         ("offset", str(shard_off)), ("length", str(len(data)))):
            e = t.external_data.add()
            e.key, e.value = key, val
        shard_off += len(data)
    sf.close()
    for h in handles.values():
        h.close()

    onnx.save_model(m, os.path.join(out_dir, "model_q4.onnx"), save_as_external_data=False)
    with open(os.path.join(out_dir, "external_data_manifest.json"), "w") as f:
        json.dump({"model": "model_q4.onnx", "shards": shard_names}, f, indent=1)

    # Caddy serves as another user; onnx/0600 outputs would 403. Make world-readable.
    for fn in ["model_q4.onnx", "external_data_manifest.json"] + shard_names:
        os.chmod(os.path.join(out_dir, fn), 0o644)
    for fn in ["model_q4.onnx"] + shard_names:
        print(f"  {fn}  {os.path.getsize(os.path.join(out_dir, fn)) / 1e6:.1f} MB")
    return emb.name, part_names, (loc, off, length)


def verify(src_dir, out_dir, emb_name, part_names, src_loc):
    """onnx.checker + prove the embedding reconstructs byte-identically."""
    onnx.checker.check_model(os.path.join(out_dir, "model_q4.onnx"))
    print("onnx.checker: OK")

    loc, off, length = src_loc
    with open(os.path.join(src_dir, loc), "rb") as f:
        f.seek(off)
        old = np.frombuffer(f.read(length), dtype=np.float16)

    new_graph = onnx.load(os.path.join(out_dir, "model_q4.onnx"), load_external_data=False)
    ninit = {t.name: t for t in new_graph.graph.initializer}
    cols = []
    for name in part_names:
        l, o, n = _ext(ninit[name])
        with open(os.path.join(out_dir, l), "rb") as f:
            f.seek(o)
            cols.append(np.frombuffer(f.read(n), dtype=np.float16).reshape(ninit[name].dims))
    rebuilt = np.concatenate(cols, axis=1).reshape(-1)
    identical = np.array_equal(old.view(np.uint16), rebuilt.view(np.uint16))
    print(f"embedding lossless (byte-identical): {identical}")
    if not identical:
        raise SystemExit("VERIFY FAILED: reconstructed embedding differs from original")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=4, help="split factor (default 4)")
    ap.add_argument("--max-shard", type=int, default=1_900_000_000,
                    help="max shard bytes, < 2 GiB V8 ArrayBuffer cap (default 1.9e9)")
    ap.add_argument("--no-verify", action="store_true", help="skip the lossless check")
    a = ap.parse_args()

    emb_name, part_names, src_loc = split(a.src_dir, a.out_dir, a.n, a.max_shard)
    if not a.no_verify:
        verify(a.src_dir, a.out_dir, emb_name, part_names, src_loc)
    print("DONE")


if __name__ == "__main__":
    main()
