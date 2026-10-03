import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
model_columns = pickle.load(open('columns.pkl', 'rb'))

st.title("🚗 Araba Fiyat Tahmin Uygulaması 🚗")
st.write("Aracın özelliklerini girerek tahmini piyasa fiyatını öğrenin!")

name = st.text_input("Araba Adı / Modeli (Örn:Maruti Swift Dzire) ",value="Maruti Swift VDI")
year = st.number_input("Aracın Üretim Yıllı", min_value=1990, max_value=2026, value=2015)
km_driven = st.number_input("Yapılan Kilometre (KM)", min_value=0, max_value=500000, value=50000)
engine = st.number_input("Motor Hacmi (CC)",min_value=500, max_value=5000, value=1248)

fuel = st.selectbox("Yakıt Türü", ["Diesel","Petrol","CNG","LPG"])
seller_type = st.selectbox("Satıcı Türü", ["Individual","Dealer","Trustmark Dealer"])
transmission = st.selectbox("Vites Tipi",["Manual","Automatic"])
owner = st.selectbox("Kaçıncı El (Owner)", ["First Owner","Second Owner","Third Owner","Fourth & Above  Owner","Test Drive Car"])


if st.button("Fiyatı Tahmin Et"):
    input_data = pd.DataFrame([[year, km_driven, engine, fuel, transmission, owner, seller_type]],
                              columns=['year', 'km_driven', 'engine', 'fuel', 'transmission','owner','seller_type'])
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns = model_columns, fill_value=0)

    prediction = model.predict(input_data)
    st.success(f"Tahmini Araba Fiyatı: {prediction[0]:,.2f} TL")
