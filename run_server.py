from flask import Flask
import threading
import bot  # Импортируем твой bot.py как модуль

app = Flask(__name__)

@app.route('/')
def index():
    return 'Bot is running.'

def start_telegram_bot():
    bot.main()

if __name__ == '__main__':
    threading.Thread(target=start_telegram_bot).start()
    app.run(host='0.0.0.0', port=10000)
