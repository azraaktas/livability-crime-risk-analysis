import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def predict_next_month(community_area_name):
    # 1. Veriyi yükle
    full_data = pd.read_csv("data/processed/community_safety_scores.csv") 
    
    # 2. İsim -> ID sözlüğü (app.py ile aynı olmalı)
    community_names = {
        1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square", 5: "North Center",
        6: "Lake View", 7: "Lincoln Park", 8: "Near North Side", 9: "Edison Park", 10: "Norwood Park",
        11: "Jefferson Park", 12: "Forest Glen", 13: "North Park", 14: "Albany Park", 15: "Portage Park",
        16: "Irving Park", 17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
        21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town", 25: "Austin",
        26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side", 29: "North Lawndale",
        30: "South Lawndale", 31: "Lower West Side", 32: "Loop", 33: "Near South Side", 34: "Armour Square",
        35: "Douglas", 36: "Oakland", 37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood",
        40: "Washington Park", 41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
        45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights", 49: "Roseland",
        50: "Pullman", 51: "South Deering", 52: "East Side", 53: "West Pullman", 54: "Riverdale",
        55: "Hegewisch", 56: "Garfield Ridge", 57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park",
        60: "Bridgeport", 61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing",
        65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood", 69: "Greater Grand Crossing",
        70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly", 73: "Washington Heights", 74: "Mount Greenwood",
        75: "Morgan Park", 76: "O'Hare", 77: "Edgewater"
    }
    
    # İsimden ID'ye ters sözlük oluştur (Rogers Park -> 1)
    name_to_id = {v: k for k, v in community_names.items()}
    area_id = name_to_id[community_area_name]
    
    # 3. Model ve Scaler'ı yükle
    model = load_model("outputs/lstm_model.keras")
    with open("outputs/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    # 4. İlgili bölgeyi ID ile filtrele
    area_data = full_data[full_data["community_area"] == area_id].sort_values("year_month")
    
    # 5. Son 3 ayı al
    last_3_months = area_data["safety_score"].values[-3:]
    
    # 6. Ölçeklendir ve tahmin et
    scaled_data = scaler.transform(last_3_months.reshape(-1, 1))
    input_data = scaled_data.reshape(1, 3, 1)
    
    prediction_scaled = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return float(prediction[0][0])