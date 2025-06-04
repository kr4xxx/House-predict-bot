import logging
import pickle
import pandas as pd
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка модели
with open('xgb_model_package.pkl', 'rb') as f:
    model_data = pickle.load(f)

KEY_RATE = 21

apartment_type_mapping = {
    '🏢 Студия': 0,
    '1️⃣ 1-комнатная': 1,
    '2️⃣ 2-комнатная': 2,
    '3️⃣ 3-комнатная': 3,
    '4️⃣ 4 и более комнат': 4
}

districts_name_to_num = {
    'Центр 🏙️': 36, 'Первая речка 🌉': 35, 'Патрокл 🌅': 34, 'Эгершельд 🌊': 33,
    'Некрасовская 🏞️': 32, 'Толстого (Буссе) 🌄': 31, 'Третья рабочая ⚙️': 30,
    'Снеговая падь ❄️': 29, 'Седанка 🌲': 28, 'Заря 🌇': 27, 'Столетие 📍': 26,
    'Чуркин 🌁': 25, 'Трудовое 🏗️': 24, 'Фадеева 🚧': 23, 'Вторая речка 🛤️': 22,
    'БАМ 🚧': 21, 'Садгород 🌿': 20, 'Чайка 🐦': 19, 'Океанская 🌊': 18,
    'Гайдамак 🏘️': 17, 'Баляева 🧭': 16, '64, 71 микрорайоны 🏢': 15,
    'Луговая 🌾': 14, 'Тихая 🤫': 13, 'Снеговая ❄️': 12, 'Сахарный ключ 🍬': 11,
    'Спутник 🛰️': 10, 'Борисенко 🛠️': 9, 'Трудовая 🧱': 8, 'Первореченский 📍': 7,
    'Весенняя 🌸': 6, 'Пригород 🏞️': 5, 'Попова 🏝️': 4, 'Горностай 🐿️': 3,
    'Русский 🏝️': 2, 'По-ов, Песчанный 🏖️': 1
}

SELECT_DISTRICT, INPUT_AREA, SELECT_APTYPE, INPUT_CURRENT_FLOOR, INPUT_TOTAL_FLOORS = range(5)

def build_keyboard(options, row_width=2):
    keyboard = []
    row = []
    for i, option in enumerate(options, 1):
        row.append(InlineKeyboardButton(option, callback_data=option))
        if i % row_width == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)

def get_main_menu():
    keyboard = [
        [InlineKeyboardButton("💰 Оценить квартиру", callback_data='estimate')],
        [InlineKeyboardButton("ℹ️ О нас", callback_data='about')],
        [InlineKeyboardButton("❤️ Поддержать проект", callback_data='support')]
    ]
    return InlineKeyboardMarkup(keyboard)

async def type_and_send(update, context, text, **kwargs):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await asyncio.sleep(0.6)
    await update.message.reply_text(text, **kwargs)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("👋 *Привет! Я помогу оценить квартиру во Владивостоке.*\n\nВыберите действие:",
                                    reply_markup=get_main_menu(), parse_mode=ParseMode.MARKDOWN)

async def main_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'estimate':
        keyboard = build_keyboard(list(districts_name_to_num.keys()), row_width=2)
        await query.edit_message_text("🌍 *Выберите район квартиры:*", reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        return SELECT_DISTRICT

    elif query.data == 'about':
        await query.edit_message_text("🏠 Мы — сервис оценки недвижимости во Владивостоке с помощью ML. Просто, быстро и бесплатно!",
                                      reply_markup=get_main_menu())

    elif query.data == 'support':
        await query.edit_message_text("🙏 Поддержите проект переводом на карту: 1234 1234 1234\n\nСпасибо за вашу помощь ❤️",
                                      reply_markup=get_main_menu())

async def select_district(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    district_name = query.data
    district_num = districts_name_to_num.get(district_name)

    if district_num is None:
        await query.edit_message_text("❌ Неверный выбор района. Попробуйте ещё раз.")
        return SELECT_DISTRICT

    context.user_data['district_num'] = district_num
    context.user_data['district_name'] = district_name

    await query.edit_message_text(f"📍 Выбран район: *{district_name}*\n\nВведите площадь квартиры (м²):", parse_mode=ParseMode.MARKDOWN)
    return INPUT_AREA

async def input_area(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        area = float(update.message.text.replace(',', '.'))
        if area <= 0:
            raise ValueError
    except ValueError:
        return await type_and_send(update, context, "⚠️ Введите положительное число для площади.")

    context.user_data['area'] = area
    keyboard = build_keyboard(list(apartment_type_mapping.keys()), row_width=2)
    await type_and_send(update, context, "🏘️ *Выберите тип квартиры:*", reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    return SELECT_APTYPE

async def select_aptype(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    aptype_name = query.data
    aptype_num = apartment_type_mapping.get(aptype_name)

    if aptype_num is None:
        await query.edit_message_text("❌ Неверный тип. Попробуйте снова.")
        return SELECT_APTYPE

    context.user_data['aptype_num'] = aptype_num
    context.user_data['aptype_name'] = aptype_name

    await query.edit_message_text("🏢 Введите этаж квартиры:")
    return INPUT_CURRENT_FLOOR

async def input_current_floor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cf = int(update.message.text)
        if cf <= 0:
            raise ValueError
    except ValueError:
        return await type_and_send(update, context, "⚠️ Введите положительное целое число для этажа.")

    context.user_data['current_floor'] = cf
    await type_and_send(update, context, "🏗️ Введите общее количество этажей в доме:")
    return INPUT_TOTAL_FLOORS

async def input_total_floors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tf = int(update.message.text)
        cf = context.user_data['current_floor']
        if tf < cf or tf <= 0:
            return await type_and_send(update, context, "⚠️ Общее количество этажей должно быть больше или равно текущему.")
    except ValueError:
        return await type_and_send(update, context, "⚠️ Введите положительное целое число для этажности.")

    context.user_data['total_floors'] = tf

    price = predict_price(
        area=context.user_data['area'],
        aptype=context.user_data['aptype_num'],
        cf=context.user_data['current_floor'],
        tf=context.user_data['total_floors'],
        key=KEY_RATE,
        district_number=context.user_data['district_num']
    )

    deviation = int(price * 0.08)
    price_str = f"{int(price):,}".replace(',', ' ')
    deviation_str = f"{deviation:,}".replace(',', ' ')

    await type_and_send(update, context,
        f"💰 *Оценка стоимости:* *{price_str} ₽*\n± {deviation_str} ₽",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_menu())
    return ConversationHandler.END

def predict_price(area, aptype, cf, tf, key, district_number):
    district_desc = model_data['districts'][district_number]
    new_data = pd.DataFrame({
        'monster': [cf / tf * 100],
        'Area': [area],
        'ApartmentType': [aptype],
        'CurrentFloor': [cf],
        'TotalFloors': [tf],
        'KeyRate': [key],
        'DistrictDesc': [district_desc]
    })

    district_encoded = pd.DataFrame(model_data['encoder'].transform(new_data[['DistrictDesc']]))
    district_encoded.columns = model_data['encoder'].get_feature_names_out(['DistrictDesc'])

    new_data = pd.concat([new_data.drop(['DistrictDesc'], axis=1), district_encoded], axis=1)
    new_data = new_data[model_data['features_order']]

    return model_data['model'].predict(new_data)[0]

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await type_and_send(update, context, "❌ Оценка отменена. Напишите /start, чтобы начать заново.")
    return ConversationHandler.END

def main():
    TOKEN = '7497598617:AAGMYwmDM2lyXhFGb_DaJisyByB7EtbuadA'
    app = ApplicationBuilder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(main_menu_handler, pattern='^estimate$')],
        states={
            SELECT_DISTRICT: [CallbackQueryHandler(select_district)],
            INPUT_AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_area)],
            SELECT_APTYPE: [CallbackQueryHandler(select_aptype)],
            INPUT_CURRENT_FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_current_floor)],
            INPUT_TOTAL_FLOORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_total_floors)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        allow_reentry=True
    )

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CallbackQueryHandler(main_menu_handler, pattern='^(about|support)$'))
    app.add_handler(conv_handler)

    print("🚀 Бот запущен...")
    app.run_polling()

if __name__ == '__main__':
    main()