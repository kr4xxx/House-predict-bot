import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
import pickle

# Загрузка данных
data = pd.read_csv('boom4.csv', sep=';', decimal=',')
data = data.dropna(axis=0)

# Удаление ненужного столбца
data = data.drop(columns=['AddressOnly'])

# Целевая переменная и признаки
y = data['Price']
data['monster'] = data['CurrentFloor'] / data['TotalFloors'] * 100
features = ['monster', 'Area', 'ApartmentType', 'CurrentFloor', 'TotalFloors', 'KeyRate', 'DistrictDesc']
X = data[features]

# Кодирование категориального признака DistrictDesc
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
district_encoded = pd.DataFrame(encoder.fit_transform(X[['DistrictDesc']]))
district_encoded.columns = encoder.get_feature_names_out(['DistrictDesc'])

# Объединение кодированных данных с основными признаками
X = pd.concat([X.drop(['DistrictDesc'], axis=1), district_encoded], axis=1)

# Разделение данных на обучающую и валидационную выборки
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Настройка модели XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 20,
    'n_estimators': 150,
    'subsample': 0.8,
    'random_state': 18
}
my_model = XGBRegressor(**xgb_params)

# Обучение модели
my_model.fit(train_X, train_y)
val_predictions = my_model.predict(val_X)
mape = mean_absolute_percentage_error(val_predictions, val_y)

# Словарь районов
districts = {
    1: 'По-ов, Песчанный', 2: 'о, Русский', 3: 'Горностай', 4: 'о, Попова',
    5: 'Пригород', 6: 'Весенняя', 7: 'Первореченский', 8: 'Трудовая',
    9: 'Борисенко', 10: 'Спутник', 11: 'Сахарный ключ', 12: 'Снеговая',
    13: 'Тихая', 14: 'Луговая', 15: '64, 71 микрорайоны', 16: 'Баляева',
    17: 'Гайдамак', 18: 'Океанская', 19: 'Чайка', 20: 'Садгород',
    21: 'БАМ', 22: 'Вторая речка', 23: 'Фадеева', 24: 'Трудовое',
    25: 'Чуркин', 26: 'Столетие', 27: 'Заря', 28: 'Седанка',
    29: 'Снеговая падь', 30: 'Третья рабочая', 31: 'Толстого (Буссе)',
    32: 'Некрасовская', 33: 'Эгершельд', 34: 'Патрокл', 35: 'Первая речка',
    36: 'Центр'
}

# Сохранение модели и необходимых объектов
model_data = {
    'model': my_model,
    'encoder': encoder,
    'districts': districts,
    'mape': mape,
    'features_order': X.columns.tolist()  # сохраняем порядок признаков
}

# Сохраняем все в один файл
with open('xgb_model_package.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Модель и вспомогательные объекты успешно сохранены в файл 'xgb_model_package.pkl'")
print(f"MAPE модели на валидационной выборке: {mape:.2%}")