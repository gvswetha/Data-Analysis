import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st 

def predict(): 
   
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    data = pd.read_csv('car_streamlit\Car details v3 (1).csv' ,encoding= 'latin 1')
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

    # Label encode categorical columns
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

    # --- Streamlit UI ---
    st.title("🚗 Car Price Prediction App")
    st.markdown("### Enter Car Details to Predict Price")

    # User input widgets
    mileage = st.number_input("Mileage (kmpl):", min_value=5.0, max_value=50.0, value=18.0, step=0.1)
    kms_driven = st.number_input("KMs Driven:", min_value=0, max_value=300000, value=20000, step=1000)
    owner = st.selectbox("Owner", options=[0, 1, 2, 3, 4],
                         format_func=lambda x: ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"][x])
    fuel_type_input = st.selectbox("Fuel Type", options=[0, 1, 2, 3],
                                   format_func=lambda x: ["CNG", "Diesel", "LPG", "Petrol"][x])
    seller_type_input = st.selectbox("Seller Type", options=[0, 1, 2],
                                     format_func=lambda x: ["Dealer", "Individual", "Trustmark Dealer"][x])
    transmission_input = st.selectbox("Transmission", options=[0, 1],
                                      format_func=lambda x: ["Automatic", "Manual"][x])

    # Predict button
    if st.button("🔍 Predict Price"):
        # Create input dictionary
        input_data = dict.fromkeys(x.columns, 0)
        input_data['km_driven'] = kms_driven
        input_data['mileage'] = mileage
        input_data['owner'] = owner

        # Set dummy flags
        if f'fuel_{fuel_type_input}' in input_data:
            input_data[f'fuel_{fuel_type_input}'] = 1
        if f'seller_type_{seller_type_input}' in input_data:
            input_data[f'seller_type_{seller_type_input}'] = 1
        if f'transmission_{transmission_input}' in input_data:
            input_data[f'transmission_{transmission_input}'] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=x.columns)

        # Make prediction
        predict_price = model.predict(input_df)[0]
        st.success(f"💰 Predicted Selling Price: ₹{predict_price:,.2f}")


        # Optional: show model input details
        with st.expander("Show model input details"):
            st.write(input_df)
            
        if predict_price is not None:
            return predict_price
        else:
            return 0  # or None

