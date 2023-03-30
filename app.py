import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("House Price Predictor")

st.write("Enter the features of the house to predict its price:")

sqft = st.number_input("Square footage:", min_value=500, max_value=10000, step=50, key="sqft")
bedrooms = st.number_input("Number of bedrooms:", min_value=1, max_value=20, step=1, key="bedrooms")
bathrooms = st.number_input("Number of bathrooms:", min_value=1, max_value=20, step=1, key="bathrooms")

uploaded_file = st.file_uploader("Upload a CSV file containing housing data:", type=["csv"])

@st.cache
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

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

if uploaded_file is not None:
    df = load_data(uploaded_file)
    model, mse, r2 = train_model(df)

    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r2)

    if st.button("Predict Price", key="predict_button"):
        input_features = np.array([sqft, bedrooms, bathrooms]).reshape(1, -1)
        prediction = model.predict(input_features)
        st.write("Predicted price:", round(prediction[0], 2))
else:
    st.write("Please upload a CSV file to get started.")
