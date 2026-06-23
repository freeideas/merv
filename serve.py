#!/usr/bin/env uvrun
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "llama-cpp-python",
# ]
# ///
"""
Model-switching proxy for Mervin/Mervis chatbot.
Uses llama-cpp-python directly -- no external llama-server binary needed.
"""

import sys
import os
import json
import signal
import threading
import re
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime, timezone

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from llama_cpp import Llama

PORT = 52836
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    'phi': os.path.join(BASE_DIR, 'phi4mini', 'model-q4_k_m.gguf'),
    'gemma': os.path.join(BASE_DIR, 'gemma4e4b', 'model-q4_k_m.gguf'),
}

llm = None
current_model = None
model_lock = threading.Lock()


def load_model(key):
    global llm, current_model
    path = MODELS[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"[serve] Loading {key} from {path}...", flush=True)
    # NOTE: this box is shared/contended (other tenants keep load avg ~6-7) and
    # llama.cpp worker threads busy-wait, so oversubscribing the CPU thrashes.
    # Measured gemma q4 throughput by thread count on this host:
    #   1t=1.37  2t=1.41  3t=1.17  4t=0.87  6t=0.48  8t=0.20  tok/s
    # Fewer threads is dramatically faster here -- 2 is the sweet spot.
    n_threads = int(os.environ.get('MERV_THREADS', '2'))
    llm = Llama(
        model_path=path,
        n_ctx=2048,
        n_threads=n_threads,
        n_threads_batch=n_threads,
        verbose=False,
    )
    current_model = key
    print(f"[serve] {key} ready", flush=True)


def do_chat_completion(messages, max_tokens=256, temperature=0.7, top_p=0.9, stream=False):
    """Run chat completion using llama-cpp-python's built-in chat handler."""
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
    )


##############################################################################
# REQUEST/RESPONSE LOG
##############################################################################

LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
_log_lock = threading.Lock()


def log_request(ip, method, path, request_body=None, response_body=None):
    now = datetime.now(timezone.utc)
    filename = now.strftime('%Y-%m-%d-%HZ.log')
    timestamp = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    entry = {
        'ts': timestamp,
        'ip': ip,
        'method': method,
        'path': path,
    }
    if request_body is not None:
        entry['request'] = request_body
    if response_body is not None:
        entry['response'] = response_body
    line = json.dumps(entry, ensure_ascii=False) + '\n'
    with _log_lock:
        with open(os.path.join(LOG_DIR, filename), 'a', encoding='utf-8') as f:
            f.write(line)


##############################################################################
# Tag fix kludge
##############################################################################

FALLBACK_RESPONSE = (
    "<Mervin>I am feeling too sad to respond right now.</Mervin>"
    "<Mervis>I am so joyful I can barely speak right now!</Mervis>"
)


def kludge_fix_tags(text):
    text = re.sub(r'<M{2,}ervin[^a-zA-Z0-9>]*>?', '<Mervin>', text)
    text = re.sub(r'<M{2,}ervis[^a-zA-Z0-9>]*>?', '<Mervis>', text)
    text = re.sub(r'<Mervin[^a-zA-Z0-9>]+>', '<Mervin>', text)
    text = re.sub(r'<Mervis[^a-zA-Z0-9>]+>', '<Mervis>', text)
    text = re.sub(r'<Mervin(?=[A-Z])', '<Mervin>', text)
    text = re.sub(r'<Mervis(?=[A-Z])', '<Mervis>', text)
    text = re.sub(r'</M+ervin[^a-zA-Z0-9>]*>', '</Mervin>', text)
    text = re.sub(r'</M+ervis[^a-zA-Z0-9>]*>', '</Mervis>', text)
    return text


def kludge_has_valid_tags(text):
    return bool(
        re.search(r'<Mervin>.*?</Mervin>', text, re.DOTALL)
        and re.search(r'<Mervis>.*?</Mervis>', text, re.DOTALL)
    )


def kludge_clean_messages(messages):
    cleaned = []
    for msg in messages:
        if msg.get('role') == 'assistant':
            content = kludge_fix_tags(msg['content'])
            if not kludge_has_valid_tags(content):
                content = FALLBACK_RESPONSE
            cleaned.append({**msg, 'content': content})
        else:
            cleaned.append(msg)
    return cleaned


##############################################################################
# HTTP handler
##############################################################################

class ProxyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            if current_model:
                self._json_response({'status': 'ok', 'model': current_model})
            else:
                self._json_response({'status': 'loading'})
        elif self.path == '/v1/models':
            models_list = [{"id": k, "object": "model"} for k in MODELS if os.path.exists(MODELS[k])]
            self._json_response({"object": "list", "data": models_list})
        elif self.path == '/':
            self.path = '/index.html'
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len)

        if self.path == '/switch':
            self._handle_switch(body)
        elif self.path == '/v1/chat/completions':
            self._handle_chat(body)
        else:
            self.send_error(404)

    def _handle_switch(self, body):
        global llm, current_model
        ip = self.client_address[0]
        try:
            data = json.loads(body)
            key = data.get('model')
            if key not in MODELS:
                self._json_response({'error': f'Unknown model: {key}'}, 400)
                return
            if key == current_model:
                self._json_response({'status': 'ok', 'model': key})
                return
            if not os.path.exists(MODELS[key]):
                self._json_response({'error': f'Model file not available: {key}'}, 404)
                return
            log_request(ip, 'POST', '/switch', request_body=f'switch to {key}')
            if not model_lock.acquire(timeout=300):
                self._json_response({'error': 'Server busy, try again'}, 503)
                return
            try:
                del llm
                llm = None
                load_model(key)
            finally:
                model_lock.release()
            self._json_response({'status': 'ok', 'model': key})
        except Exception as e:
            self._json_response({'error': str(e)}, 500)

    def _handle_chat(self, body):
        ip = self.client_address[0]
        if not current_model or llm is None:
            self._json_response({'error': 'Model still loading'}, 503)
            return

        try:
            req_data = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({'error': 'Invalid JSON'}, 400)
            return

        messages = req_data.get('messages', [])
        messages = kludge_clean_messages(messages)
        max_tokens = req_data.get('max_tokens', 256)
        temperature = req_data.get('temperature', 0.7)
        top_p = req_data.get('top_p', 0.9)
        stream = req_data.get('stream', False)

        user_msgs = [m['content'] for m in messages if m.get('role') == 'user']
        log_user_msg = user_msgs[-1] if user_msgs else None

        if not model_lock.acquire(timeout=300):
            self._json_response({'error': 'Server busy, try again'}, 503)
            return

        try:
            if stream:
                self._handle_stream(messages, max_tokens, temperature, top_p, ip, log_user_msg)
            else:
                result = do_chat_completion(messages, max_tokens, temperature, top_p, stream=False)
                response_text = result['choices'][0]['message']['content']
                log_request(ip, 'POST', '/v1/chat/completions',
                            request_body=log_user_msg, response_body=response_text)
                self._json_response(result)
        except Exception as e:
            log_request(ip, 'POST', '/v1/chat/completions',
                        request_body=log_user_msg, response_body=f'ERROR: {e}')
            self._json_response({'error': str(e)}, 500)
        finally:
            model_lock.release()

    def _handle_stream(self, messages, max_tokens, temperature, top_p, ip, log_user_msg):
        self.send_response(200)
        self._cors_headers()
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()

        full_content = ''
        try:
            for chunk in do_chat_completion(messages, max_tokens, temperature, top_p, stream=True):
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')
                if content:
                    full_content += content
                line = f"data: {json.dumps(chunk)}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as e:
            full_content += f' [ERROR: {e}]'

        log_request(ip, 'POST', '/v1/chat/completions',
                    request_body=log_user_msg, response_body=full_content)

    def _json_response(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self._cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')


class ThreadedHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


##############################################################################
# Entry point
##############################################################################

def main():
    first_model = None
    for key, path in MODELS.items():
        if os.path.exists(path):
            first_model = key
            break
    if not first_model:
        print("ERROR: No model files found!")
        sys.exit(1)

    print(f"[serve] Loading first model: {first_model}...", flush=True)
    load_model(first_model)

    server = ThreadedHTTPServer(('127.0.0.1', PORT), ProxyHandler)
    print(f"[serve] Listening on http://localhost:{PORT}", flush=True)
    print(f"[serve] Available models: {list(MODELS.keys())}", flush=True)

    def cleanup(*_):
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cleanup()


if __name__ == '__main__':
    main()
