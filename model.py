import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
df = pd.read_csv("car_data.csv")
df = df.dropna()

if 'engine' in df.columns and df['engine'].dtype == 'object':
    df['engine'] = df['engine'].str.replace('CC','',regex=False).astype(float)

x = df[['year', 'km_driven','engine','fuel','transmission','owner','seller_type']]
x = pd.get_dummies(x, drop_first=True)

y = df['selling_price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)

print(f"Model Başarı Skoru (Accuracy): {model.score(x_test, y_test)}")

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('columns.pkl','wb') as file:
    pickle.dump(x.columns, file)

print("Yeni Model ve Sütunlar başarıyla kaydedildi!")

