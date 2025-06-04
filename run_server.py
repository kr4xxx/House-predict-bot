import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import bot  # Импорт твоего bot.py с функцией main()

# Простой HTTP-сервер, чтобы Render "видел порт"
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')

def run_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    server.serve_forever()

# Запускаем Telegram-бота и HTTP-сервер в отдельных потоках
threading.Thread(target=bot.main).start()
run_http_server()
