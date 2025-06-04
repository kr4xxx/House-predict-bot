"""
Telegram bot for apartment price estimation with built‑in keep‑alive pings so a free Render instance
won't fall asleep after 15 min of inactivity. Just deploy this script, make sure that the environment
variable RENDER_EXTERNAL_HOSTNAME is present (Render sets it automatically), and add `aiohttp`
to your requirements.txt.

Main additions compared to the original version:
  • `import aiohttp` and `keep_alive()` coroutine (sends GET to https://<RENDER_EXTERNAL_HOSTNAME>/ every 9 min).
  • `asyncio.create_task(keep_alive())` inside the `run()` startup function.
  • TOKEN can now be supplied via the `TELEGRAM_BOT_TOKEN` env var (hard‑coded fallback kept for dev).
"""

import logging
import pickle
import os
import asyncio
import aiohttp  # NEW: for keep‑alive pings

import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# --------------------------------------------------
# Logger setup
# --------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load ML model package (dict with: model, encoder, districts, features_order)
# --------------------------------------------------
with open("xgb_model_package.pkl", "rb") as f:
    model_data = pickle.load(f)

KEY_RATE = 21  # ЦБ ключевая ставка на дату обучения модели

# --------------------------------------------------
# Constant mappings
# --------------------------------------------------

apartment_type_mapping = {
    "🏢 Студия": 0,
    "1️⃣ 1-комнатная": 1,
    "2️⃣ 2-комнатная": 2,
    "3️⃣ 3-комнатная": 3,
    "4️⃣ 4 и более комнат": 4,
}

districts_name_to_num = {
    "Центр 🏙️": 36,
    "Первая речка 🌉": 35,
    "Патрокл 🌅": 34,
    "Эгершельд 🌊": 33,
    "Некрасовская 🏞️": 32,
    "Толстого (Буссе) 🌄": 31,
    "Третья рабочая ⚙️": 30,
    "Снеговая падь ❄️": 29,
    "Седанка 🌲": 28,
    "Заря 🌇": 27,
    "Столетие 📍": 26,
    "Чуркин 🌁": 25,
    "Трудовое 🏗️": 24,
    "Фадеева 🚧": 23,
    "Вторая речка 🛤️": 22,
    "БАМ 🚧": 21,
    "Садгород 🌿": 20,
    "Чайка 🐦": 19,
    "Океанская 🌊": 18,
    "Гайдамак 🏘️": 17,
    "Баляева 🧭": 16,
    "64, 71 микрорайоны 🏢": 15,
    "Луговая 🌾": 14,
    "Тихая 🤫": 13,
    "Снеговая ❄️": 12,
    "Сахарный ключ 🍬": 11,
    "Спутник 🛰️": 10,
    "Борисенко 🛠️": 9,
    "Трудовая 🧱": 8,
    "Первореченский 📍": 7,
    "Весенняя 🌸": 6,
    "Пригород 🏞️": 5,
    "Попова 🏝️": 4,
    "Горностай 🐿️": 3,
    "Русский 🏝️": 2,
    "По-ов, Песчанный 🏖️": 1,
}

# Conversation state constants
SELECT_DISTRICT, INPUT_AREA, SELECT_APTYPE, INPUT_CURRENT_FLOOR, INPUT_TOTAL_FLOORS = range(5)

# --------------------------------------------------
# Helper UI builders
# --------------------------------------------------

def build_keyboard(options, row_width: int = 2) -> InlineKeyboardMarkup:
    keyboard, row = [], []
    for i, option in enumerate(options, 1):
        row.append(InlineKeyboardButton(option, callback_data=option))
        if i % row_width == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)

def get_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💰 Оценить квартиру", callback_data="estimate")],
        [InlineKeyboardButton("ℹ️ О нас", callback_data="about")],
        [InlineKeyboardButton("❤️ Поддержать проект", callback_data="support")],
    ])

# --------------------------------------------------
# Bot message helpers
# --------------------------------------------------
async def type_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await asyncio.sleep(0.6)
    await update.message.reply_text(text, **kwargs)

# --------------------------------------------------
# Command / start + main menu
# --------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "👋 *Привет! Я помогу оценить квартиру во Владивостоке.*\n\nВыберите действие:",
        reply_markup=get_main_menu(),
        parse_mode=ParseMode.MARKDOWN,
    )

async def main_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "estimate":
        keyboard = build_keyboard(list(districts_name_to_num.keys()))
        await query.edit_message_text(
            "🌍 *Выберите район квартиры:*",
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN,
        )
        return SELECT_DISTRICT
    elif query.data == "about":
        await query.edit_message_text(
            "🏠 Мы — сервис оценки недвижимости во Владивостоке с помощью ML. Просто, быстро и бесплатно!",
            reply_markup=get_main_menu(),
        )
    elif query.data == "support":
        await query.edit_message_text(
            "🙏 Поддержите проект переводом на карту: +79241379584\n\nСпасибо за вашу помощь ❤️",
            reply_markup=get_main_menu(),
        )

# --------------------------------------------------
# Conversation flow handlers
# --------------------------------------------------
async def select_district(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    district_name = query.data
    district_num = districts_name_to_num.get(district_name)
    if district_num is None:
        await query.edit_message_text("❌ Неверный выбор района. Попробуйте ещё раз.")
        return SELECT_DISTRICT
    context.user_data["district_num"] = district_num
    context.user_data["district_name"] = district_name

    await query.edit_message_text(
        f"📍 Выбран район: *{district_name}*\n\nВведите площадь квартиры (м²):",
        parse_mode=ParseMode.MARKDOWN,
    )
    return INPUT_AREA

async def input_area(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        area = float(update.message.text.replace(",", "."))
        if area <= 0:
            raise ValueError
    except ValueError:
        return await type_and_send(update, context, "⚠️ Введите положительное число для площади.")

    context.user_data["area"] = area
    keyboard = build_keyboard(list(apartment_type_mapping.keys()))
    await type_and_send(
        update,
        context,
        "🏘️ *Выберите тип квартиры:*",
        reply_markup=keyboard,
        parse_mode=ParseMode.MARKDOWN,
    )
    return SELECT_APTYPE

async def select_aptype(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    aptype_name = query.data
    aptype_num = apartment_type_mapping.get(aptype_name)
    if aptype_num is None:
        await query.edit_message_text("❌ Неверный тип. Попробуйте снова.")
        return SELECT_APTYPE

    context.user_data["aptype_num"] = aptype_num
    context.user_data["aptype_name"] = aptype_name

    await query.edit_message_text("🏢 Введите этаж квартиры:")
    return INPUT_CURRENT_FLOOR

async def input_current_floor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cf = int(update.message.text)
        if cf <= 0:
            raise ValueError
    except ValueError:
        return await type_and_send(update, context, "⚠️ Введите положительное целое число для этажа.")

    context.user_data["current_floor"] = cf
    await type_and_send(update, context, "🏗️ Введите общее количество этажей в доме:")
    return INPUT_TOTAL_FLOORS

async def input_total_floors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tf = int(update.message.text)
        cf = context.user_data["current_floor"]
        if tf < cf or tf <= 0:
            return await type_and_send(
                update,
                context,
                "⚠️ Общее количество этажей должно быть больше или равно текущему.",
            )
    except ValueError:
        return await type_and_send(update, context, "⚠️ Введите положительное целое число для этажности.")

    context.user_data["total_floors"] = tf

    price = predict_price(
        area=context.user_data["area"],
        aptype=context.user_data["aptype_num"],
        cf=context.user_data["current_floor"],
        tf=context.user_data["total_floors"],
        key=KEY_RATE,
        district_number=context.user_data["district_num"],
    )

    deviation = int(price * 0.08)
    price_str = f"{int(price):,}".replace(",", " ")
    deviation_str = f"{deviation:,}".replace(",", " ")

    await type_and_send(
        update,
        context,
        f"💰 *Оценка стоимости:* *{price_str} ₽*\n± {deviation_str} ₽",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_menu(),
    )
    return ConversationHandler.END

# --------------------------------------------------
# ML prediction helper
# --------------------------------------------------

def predict_price(area, aptype, cf, tf, key, district_number):
    district_desc = model_data["districts"][district_number]
    new_data = pd.DataFrame(
        {
            "monster": [cf / tf * 100],
            "Area": [area],
            "ApartmentType": [aptype],
            "CurrentFloor": [cf],
            "TotalFloors": [tf],
            "KeyRate": [key],
            "DistrictDesc": [district_desc],
        }
    )

    # One‑hot encode district
    district_encoded = pd.DataFrame(model_data["encoder"].transform(new_data[["DistrictDesc"]]))
    district_encoded.columns = model_data["encoder"].get_feature_names_out(["DistrictDesc"])

    new_data = pd.concat([new_data.drop(["DistrictDesc"], axis=1), district_encoded], axis=1)
    new_data = new_data[model_data["features_order"]]

    return model_data["model"].predict(new_data)[0]

# --------------------------------------------------
# Cancel handler
# --------------------------------------------------

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await type_and_send(update, context, "❌ Оценка отменена. Напишите /start, чтобы начать заново.")
    return ConversationHandler.END

# --------------------------------------------------
# Keep‑alive coroutine (NEW)
# --------------------------------------------------
PING_INTERVAL = 9 * 60  # 9 минут < 15‑минутного лимита Render


async def keep_alive():
    """Periodically pings the app's URL so the free Render instance stays awake."""
    # Wait a bit to ensure the webhook is set
    await asyncio.sleep(10)

    host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    if not host:
        logger.warning("RENDER_EXTERNAL_HOSTNAME is not set; keep‑alive disabled")
        return

    url = f"https://{host}/"  # root path is fine; 404 also wakes the container
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(url, timeout=10) as resp:
                    logger.info("Keep‑alive ping %s → %s", url, resp.status)
            except Exception as exc:
                logger.warning("Keep‑alive failed: %s", exc)
            await asyncio.sleep(PING_INTERVAL)

# --------------------------------------------------
# Application setup & launch
# --------------------------------------------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
application = ApplicationBuilder().token(TOKEN).build()

# Handlers
conv_handler = ConversationHandler(
    entry_points=[CallbackQueryHandler(main_menu_handler, pattern="^estimate$")],
    states={
        SELECT_DISTRICT: [CallbackQueryHandler(select_district)],
        INPUT_AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_area)],
        SELECT_APTYPE: [CallbackQueryHandler(select_aptype)],
        INPUT_CURRENT_FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_current_floor)],
        INPUT_TOTAL_FLOORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_total_floors)],
    },
    fallbacks=[CommandHandler("cancel", cancel)],
    allow_reentry=True,
)

application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(main_menu_handler, pattern="^(about|support)$"))
application.add_handler(conv_handler)


# Main entry point
async def run():
    await application.initialize()
    await application.start()

    # Set webhook
    host = os.environ["RENDER_EXTERNAL_HOSTNAME"]
    webhook_url = f"https://{host}/telegram"
    await application.bot.set_webhook(url=webhook_url)
    await application.updater.start_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        url_path="/telegram",
        webhook_url=webhook_url,
    )

    # Start keep‑alive ping task
    asyncio.create_task(keep_alive())

    logger.info("✅ Bot is up via webhook and keep‑alive is running!")


if __name__ == "__main__":
    asyncio.run(run())
