import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("House Price Predictor")

st.write("Enter the features of the house to predict its price:")

sqft = st.number_input("Square footage:", min_value=500, max_value=10000, step=50)
bedrooms = st.number_input("Number of bedrooms:", min_value=1, max_value=20, step=1)
bathrooms = st.number_input("Number of bathrooms:", min_value=1, max_value=20, step=1)

@st.cache
def load_data():
    data_url = "https://github.com/mtarief/streamlit/blob/main/housing_data.csv"
    return pd.read_csv(data_url)

def train_model(df):
    X = df[["sqft", "bedrooms", "bathrooms"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

df = load_data()
model, mse, r2 = train_model(df)

st.write("Mean Squared Error:", mse)
st.write("R-squared:", r2)

if st.button("Predict Price"):
    input_features = np.array([sqft, bedrooms, bathrooms]).reshape(1, -1)
    prediction = model.predict(input_features)
    st.write("Predicted price:", round(prediction[0], 2))

