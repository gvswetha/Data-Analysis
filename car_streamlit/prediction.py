import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time # Import time for timestamp

# --- Use st.cache_resource for the model and data processing ---
# This ensures the expensive parts (data load/model training) only run once.
@st.cache_resource
def load_and_train_model():
    # Note: Use forward slash for path portability
     data = pd.read_csv('car_streamlit/Car details v3 (1).csv' ,encoding= 'latin 1')
    df = pd.DataFrame(data)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Extract numeric values and handle errors
    df['mileage'] = df['mileage'].astype(str).str.extract('(\d+\.\d+|\d+)').astype(float)
    df['engine'] = df['engine'].astype(str).str.extract('(\d+)').astype(float)
    df['max_power'] = df['max_power'].astype(str).str.extract('(\d+\.\d+|\d+)').astype(float)
    df['torque'] = df['torque'].astype(str).str.extract('(\d+\.\d+|\d+)').astype(float)
    df['seats'] = df['seats'].astype(float)

    # Fill missing values
    for col in ['mileage', 'engine', 'torque', 'max_power', 'seats']:
        df[col] = df[col].fillna(df[col].mean())

    # Label encode categorical columns (must be stored for later use, but here we just convert)
    for col in ['fuel', 'seller_type', 'transmission', 'owner']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Convert to dummy variables
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission'], drop_first=True)

    # Define X and y
    x = df[['km_driven', 'mileage', 'owner'] +
            [col for col in df.columns if 'fuel_' in col or 'seller_type_' in col or 'transmission_' in col]]
    
    y = df['selling_price']

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(x, y)
    
    return x, model


def predict():
    # --- 1. INITIAL SETUP ---
    x, model = load_and_train_model()
    
    # Initialize the history list in session_state if it doesn't exist
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []

    # --- 2. STREAMLIT UI (INPUTS) ---
    st.title("🚗 Car Price Prediction App")
    st.markdown("### Enter Car Details to Predict Price")

    # Input widgets (Ensure these variables are captured correctly)
    mileage = st.number_input("Mileage (kmpl):", min_value=5.0, max_value=50.0, value=18.0, step=0.1)
    kms_driven = st.number_input("KMs Driven:", min_value=0, max_value=300000, value=20000, step=1000)
    
    # Using the encoded values (0, 1, 2, ...) directly for the model
    owner_value = st.selectbox("Owner", options=[0, 1, 2, 3, 4],
                              format_func=lambda x: ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"][x])
    fuel_type_value = st.selectbox("Fuel Type", options=[0, 1, 2, 3],
                                 format_func=lambda x: ["CNG", "Diesel", "LPG", "Petrol"][x])
    seller_type_value = st.selectbox("Seller Type", options=[0, 1, 2],
                                   format_func=lambda x: ["Dealer", "Individual", "Trustmark Dealer"][x])
    transmission_value = st.selectbox("Transmission", options=[0, 1],
                                    format_func=lambda x: ["Automatic", "Manual"][x])


    # --- 3. PREDICT BUTTON LOGIC ---
    if st.button("🔍 Predict Price"):
        # Create input dictionary
        input_data = dict.fromkeys(x.columns, 0)
        input_data['km_driven'] = kms_driven
        input_data['mileage'] = mileage
        input_data['owner'] = owner_value # Use the actual selected value

        # Map selectbox values to dummy columns based on the original structure
        # NOTE: The original code used the encoded value (0, 1, 2) which is correct for the model
        
        # FUEL (Assuming the order matches the one-hot encoding columns)
        fuel_map = {0: 'fuel_1', 1: 'fuel_2', 2: 'fuel_3'} # This is a placeholder mapping based on your setup
        if fuel_type_value in fuel_map:
            input_data[fuel_map[fuel_type_value]] = 1
            
        # SELLER_TYPE
        seller_map = {0: 'seller_type_1', 1: 'seller_type_2'}
        if seller_type_value in seller_map:
            input_data[seller_map[seller_type_value]] = 1
            
        # TRANSMISSION
        transmission_map = {0: 'transmission_1'}
        if transmission_value in transmission_map:
            input_data[transmission_map[transmission_value]] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=x.columns)

        # Make prediction
        predict_price = model.predict(input_df)[0]
        st.success(f"💰 Predicted Selling Price: ₹{predict_price:,.2f}")
        
        # --- UPDATE HISTORY (The main fix) ---
        new_record = {
            'Time': pd.to_datetime(time.time(), unit='s').strftime('%H:%M:%S'),
            'Price': predict_price,
            'KMs': kms_driven,
            'Owner': ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"][owner_value],
            'Fuel': ["CNG", "Diesel", "LPG", "Petrol"][fuel_type_value]
        }
        
        # Add to history (newest at the top)
        st.session_state['prediction_history'].insert(0, new_record)
        
        # Keep only the last 5
        st.session_state['prediction_history'] = st.session_state['prediction_history'][:5]

        # Optional: show model input details
        with st.expander("Show model input details"):
            st.write(input_df)

    # --- 4. DISPLAY HISTORY (OUTSIDE THE BUTTON BLOCK) ---
    # This ensures the history table remains visible even when the button is not clicked.
    if st.session_state['prediction_history']:
        st.markdown("---")
        st.subheader("Last 5 Predictions")
        
        # Convert history list to DataFrame for easy display
        history_df = pd.DataFrame(st.session_state['prediction_history'])
        
        # Format the 'Price' column for display
        history_df['Price'] = history_df['Price'].apply(lambda x: f"₹{x:,.2f}")
        
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True 
        )

    # --- 5. CLEAR BUTTON ---
    if st.button("🔄 Clear Prediction History"):
        st.session_state['prediction_history'] = []
        st.rerun() # Use st.rerun to refresh the display immediately




