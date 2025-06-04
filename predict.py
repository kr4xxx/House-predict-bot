import pickle
import pandas as pd

# Загрузка модели и вспомогательных объектов
with open('xgb_model_package.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Пример использования
def predict_price(area, aptype, cf, tf, key, district_number):
    # Получаем название района по номеру
    district_desc = model_data['districts'][district_number]
    
    # Создаем DataFrame с входными данными
    new_data = pd.DataFrame({
        'monster': [cf / tf * 100],
        'Area': [area],
        'ApartmentType': [aptype],
        'CurrentFloor': [cf],
        'TotalFloors': [tf],
        'KeyRate': [key],
        'DistrictDesc': [district_desc]
    })
    
    # Кодируем категориальный признак
    district_encoded = pd.DataFrame(model_data['encoder'].transform(new_data[['DistrictDesc']]))
    district_encoded.columns = model_data['encoder'].get_feature_names_out(['DistrictDesc'])
    
    # Объединяем с остальными признаками
    new_data = pd.concat([new_data.drop(['DistrictDesc'], axis=1), district_encoded], axis=1)
    
    # Убедимся, что порядок признаков совпадает с обучением
    new_data = new_data[model_data['features_order']]
    
    # Делаем предсказание
    prediction = model_data['model'].predict(new_data)[0]
    
    return prediction
