import asyncio
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from bot import application  # Импортируем объект Application

# HTTP-сервер, чтобы Render видел порт
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")

def run_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()

async def run_bot():
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    await application.updater.idle()

if __name__ == "__main__":
    Thread(target=run_http_server).start()
    asyncio.run(run_bot())
