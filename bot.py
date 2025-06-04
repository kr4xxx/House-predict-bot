import logging
import pickle
import asyncio
import os

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

# ─────────────────────────────
# Логирование
# ─────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────
# Загрузка пакета модели
# ─────────────────────────────
with open("xgb_model_package.pkl", "rb") as f:
    model_data = pickle.load(f)  # dict: model / encoder / districts / features_order

KEY_RATE = 21  # ключевая ставка ЦБ на момент обучения

# ─────────────────────────────
# Словари и константы
# ─────────────────────────────
apartment_type_mapping = {
    "🏢 Студия": 0,
    "1️⃣ 1-комнатная": 1,
    "2️⃣ 2-комнатная": 2,
    "3️⃣ 3-комнатная": 3,
    "4️⃣ 4 и более": 4,
}
districts_name_to_num = {
    "Центр 🏙️": 36, "Первая речка 🌉": 35, "Патрокл 🌅": 34, "Эгершельд 🌊": 33,
    "Некрасовская 🏞️": 32, "Толстого (Буссе) 🌄": 31, "Третья рабочая ⚙️": 30,
    "Снеговая падь ❄️": 29, "Седанка 🌲": 28, "Заря 🌇": 27, "Столетие 📍": 26,
    "Чуркин 🌁": 25, "Трудовое 🏗️": 24, "Фадеева 🚧": 23, "Вторая речка 🛤️": 22,
    "БАМ 🚧": 21, "Садгород 🌿": 20, "Чайка 🐦": 19, "Океанская 🌊": 18,
    "Гайдамак 🏘️": 17, "Баляева 🧭": 16, "64, 71 мкр. 🏢": 15, "Луговая 🌾": 14,
    "Тихая 🤫": 13, "Снеговая ❄️": 12, "Сахарный ключ 🍬": 11, "Спутник 🛰️": 10,
    "Борисенко 🛠️": 9, "Трудовая 🧱": 8, "Первореченский 📍": 7, "Весенняя 🌸": 6,
    "Пригород 🏞️": 5, "Попова 🏝️": 4, "Горностай 🐿️": 3, "Русский 🏝️": 2,
    "По-ов, Песчанный 🏖️": 1,
}

SELECT_DISTRICT, INPUT_AREA, SELECT_APTYPE, INPUT_CURRENT_FLOOR, INPUT_TOTAL_FLOORS = range(5)

# ─────────────────────────────
# Вспомогательные UI-функции
# ─────────────────────────────
def build_keyboard(options, row_width: int = 2) -> InlineKeyboardMarkup:
    rows, row = [], []
    for i, opt in enumerate(options, 1):
        row.append(InlineKeyboardButton(opt, callback_data=opt))
        if i % row_width == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)

def get_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("💰 Оценить квартиру", callback_data="estimate")],
            [InlineKeyboardButton("ℹ️ О нас", callback_data="about")],
            [InlineKeyboardButton("❤️ Поддержать проект", callback_data="support")],
        ]
    )

async def type_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, **kwargs):
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    await asyncio.sleep(0.6)
    if update.message:
        await update.message.reply_text(text, **kwargs)
    else:
        await update.callback_query.message.reply_text(text, **kwargs)

# ─────────────────────────────
# Обработчики команд и шагов
# ─────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "👋 *Привет! Я помогу оценить квартиру во Владивостоке.*\n\nВыберите действие:",
        reply_markup=get_main_menu(),
        parse_mode=ParseMode.MARKDOWN,
    )

async def main_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "estimate":
        await q.edit_message_text(
            "🌍 *Выберите район квартиры:*",
            reply_markup=build_keyboard(districts_name_to_num.keys()),
            parse_mode=ParseMode.MARKDOWN,
        )
        return SELECT_DISTRICT
    if q.data == "about":
        await q.edit_message_text(
            "🏠 Мы — сервис оценки недвижимости во Владивостоке, использующий ML-модель XGBoost.",
            reply_markup=get_main_menu(),
        )
    if q.data == "support":
        await q.edit_message_text(
            "🙏 Поддержите проект переводом на карту +7 924 137-95-84. Спасибо! ❤️",
            reply_markup=get_main_menu(),
        )

async def select_district(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    district_name = q.data
    district_num = districts_name_to_num.get(district_name)
    if district_num is None:
        await q.edit_message_text("❌ Неверный район. Попробуйте ещё раз.")
        return SELECT_DISTRICT
    context.user_data.update(district_num=district_num, district_name=district_name)
    await q.edit_message_text(
        f"📍 *{district_name}* выбран.\n\nВведите площадь квартиры (м²):",
        parse_mode=ParseMode.MARKDOWN,
    )
    return INPUT_AREA

async def input_area(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        area = float(update.message.text.replace(",", "."))
        if area <= 0:
            raise ValueError
    except ValueError:
        return await type_and_send(update, context, "⚠️ Введите положительное число.")
    context.user_data["area"] = area
    await type_and_send(
        update, context,
        "🏘 *Выберите тип квартиры:*",
        reply_markup=build_keyboard(apartment_type_mapping.keys()),
        parse_mode=ParseMode.MARKDOWN,
    )
    return SELECT_APTYPE

async def select_aptype(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    aptype_name = q.data
    aptype_num = apartment_type_mapping.get(aptype_name)
    if aptype_num is None:
        await q.edit_message_text("❌ Неверный тип. Попробуйте ещё раз.")
        return SELECT_APTYPE
    context.user_data.update(aptype_num=aptype_num, aptype_name=aptype_name)
    await q.edit_message_text("🏢 Введите этаж квартиры:")
    return INPUT_CURRENT_FLOOR

async def input_current_floor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cf = int(update.message.text)
        if cf <= 0:
            raise ValueError
    except ValueError:
        return await type_and_send(update, context, "⚠️ Введите целое положительное число.")
    context.user_data["current_floor"] = cf
    await type_and_send(update, context, "🏗 Введите общее количество этажей:")
    return INPUT_TOTAL_FLOORS

async def input_total_floors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tf = int(update.message.text)
        cf = context.user_data["current_floor"]
        if tf < cf or tf <= 0:
            raise ValueError
    except ValueError:
        return await type_and_send(update, context, "⚠️ Этажей должно быть ≥ текущего.")
    context.user_data["total_floors"] = tf

    price = predict_price(
        area=context.user_data["area"],
        aptype=context.user_data["aptype_num"],
        district_num=context.user_data["district_num"],
        current_floor=context.user_data["current_floor"],
        total_floors=context.user_data["total_floors"],
    )
    dev = int(price * 0.08)
    await type_and_send(
        update, context,
        f"💰 *Оценка:* {int(price):,} ₽ ± {dev:,} ₽".replace(",", " "),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_menu(),
    )
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Оценка отменена. /start — сначала.")
    return ConversationHandler.END

# ─────────────────────────────
# ML-функция предсказания
# ─────────────────────────────
def predict_price(area, aptype, district_num, current_floor, total_floors):
    district_desc = model_data["districts"][district_num]
    df = pd.DataFrame({
        "monster": [current_floor / total_floors * 100],
        "Area": [area],
        "ApartmentType": [aptype],
        "CurrentFloor": [current_floor],
        "TotalFloors": [total_floors],
        "KeyRate": [KEY_RATE],
        "DistrictDesc": [district_desc],
    })
    enc_df = pd.DataFrame(
        model_data["encoder"].transform(df[["DistrictDesc"]]),
        columns=model_data["encoder"].get_feature_names_out(["DistrictDesc"]),
    )
    df = pd.concat([df.drop(columns="DistrictDesc"), enc_df], axis=1)
    df = df[model_data["features_order"]]
    return model_data["model"].predict(df)[0]

# ─────────────────────────────
# keep-alive: дешёвый вызов к API
# ─────────────────────────────
async def keep_alive(bot):
    while True:
        try:
            await bot.get_me()
        except Exception as exc:
            logger.warning("keep-alive error: %s", exc)
        await asyncio.sleep(60 * 9)

# ─────────────────────────────
# Точка входа
# ─────────────────────────────
def main():
    TOKEN = "ВСТАВЬ_СЮДА_СВОЙ_ТОКЕН"  # ← ВСТАВЬ сюда свой токен БЕЗ os.getenv
    app = ApplicationBuilder().token(TOKEN).build()

    # Хендлеры
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(main_menu_handler)],
        states={
            SELECT_DISTRICT: [CallbackQueryHandler(select_district)],
            INPUT_AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_area)],
            SELECT_APTYPE: [CallbackQueryHandler(select_aptype)],
            INPUT_CURRENT_FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_current_floor)],
            INPUT_TOTAL_FLOORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_total_floors)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)

    # Keep-alive loop
    asyncio.get_event_loop().create_task(keep_alive(app.bot))

    app.run_polling()

# ─────────────────────────────
# Запуск
# ─────────────────────────────
if __name__ == "__main__":
    main()
