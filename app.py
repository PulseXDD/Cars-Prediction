import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('xgboost.pkl')

le_brand = joblib.load('brand.pkl')

brand_mapping = dict(zip(le_brand.classes_, le_brand.transform(le_brand.classes_)))

st.title("Prediksi Harga Mobil")

engine_capacity = st.number_input('Engine Capacity (L)', min_value=0.0, step=0.1)
cylinder = st.number_input('Jumlah Cylinder', min_value=1, step=1)
horse_power = st.number_input('Horse Power (HP)', min_value=1, step=1)
top_speed = st.number_input('Top Speed (km/h)', min_value=1, step=1)
seats = st.number_input('Jumlah Seats', min_value=1, step=1)
brand = st.selectbox('Brand', list(brand_mapping.keys()))

image_path = f'images/{brand.lower().replace(" ", "_").replace("-", "_")}.jpg'
st.image(image_path, caption=brand)

if st.button("Prediksi Harga"):
    input_data = pd.DataFrame([[
        engine_capacity,
        cylinder,
        horse_power,
        top_speed,
        seats,
        brand_mapping[brand]
    ]], columns=['engine_capacity', 'cylinder', 'horse_power', 'top_speed', 'seats', 'brand'])

    prediction = model.predict(input_data)[0]
    # Konversi SAR ke IDR
    exchange_rate = 4373  # 1 SAR = 4373 IDR
    prediction_idr = prediction * exchange_rate
    st.success(f"Prediksi harga mobil: Rp {prediction_idr:,.0f}")