import streamlit as st 
import pandas as pd 
from employ_ann import test_train,feature_eng,kpi,predict,set_bg,charts

import streamlit as st

data = pd.read_csv('HR_Attrition/HR_Attrition(streamlit)/HR-Employee-Attrition.csv',encoding="latin 1")
df = pd.DataFrame(data)

st.markdown("""
<style>


/* Sidebar background */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.25) !important; /* Black with 25% opacity */
    backdrop-filter: blur(6px); /* Optional: Glass effect */
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;         /* Force text to white */
}

/* Sidebar input boxes */
[data-testid="stSidebar"] .stTextInput>div>div>input,
[data-testid="stSidebar"] .stNumberInput>div>div>input,
[data-testid="stSidebar"] select {
    background-color: rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Dropdown text */
[data-testid="stSidebar"] .stSelectbox div div {
    color: white !important;
}
    p{
        color:white;
    }


    .st-emotion-cache-ua1rfn{
        color:white;
    }

</style>
""", unsafe_allow_html=True)

import streamlit as st

def inject_opacity_css():
    st.markdown("""
        <style>
        /* Target the main table container */
        div[data-testid="stTable"] > div {
            background-color: rgba(255, 255, 255, 0.3) !important;  /* White with 30% opacity */
            box-shadow: none !important;
            border: none !important;
            color: white !important;
        }

        /* Table and headers background transparent */
        div[data-testid="stTable"] table,
        div[data-testid="stTable"] thead th,
        div[data-testid="stTable"] tbody td {
            background-color: transparent !important;
            color: white !important;
            border: none !important;
        }

        /* Remove row hover background */
        div[data-testid="stTable"] tbody tr:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
        }
        </style>
    """, unsafe_allow_html=True)

inject_opacity_css()


set_bg("HR_Attrition/HR_Attrition(streamlit)/slider-main-slide.jpg")
st.title("👩🏻‍💻 Employee Attrition Prediction using ANN 📊")

# Employee_Attrition_Probability,history,df = test_train()

menu = st.sidebar.radio("Go to", ["Dashboard", "Attrition Prediction","data_table"])

if menu == "Dashboard":
    col1,col2,col3,col4 = kpi()
    charts()
    
elif menu == "Attrition Prediction":
    test_train(df)
    predict()
    
elif menu == "data_table":
    monthly_income_range = feature_eng()

   

