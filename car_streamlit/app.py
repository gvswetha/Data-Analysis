import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt 
from car_prediction import yearly_profit
from main import header,visual,chart3
from prediction import predict
from data_table import show_table,prepare_data,show_tb2,show_chart,chart3,generate_transmission_chart
# Inject CSS to style the DataFrame
st.markdown(
    """
    <style>
   
    /* Optional: Style the table itself if needed, although the container usually suffices */
    .stDataFrame table {
        border-radius: 10px;
    }
    .st-emotion-cache-kmufyr{
        border-radius: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.8);
        margin: 10px;
    }
   
    .st-emotion-cache-14vh5up{
       background-image: linear-gradient(to right, #f8e9ee, #e4bef7);
    }
    
    .mark-group role-title{
        align : center;
    }
    
    .chart-wrapper fit-x fit-y{
        background:transparent;
    }
    
    path.background{
        fill:"white";
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Display your styled table
 # Assuming this function calls st.dataframe
# Custom CSS to target the main app container for the background color
# Place this section immediately after your imports in app.py
st.markdown(
    """
<style>
    .st-emotion-cache-1yiq2ps{
        background-image: linear-gradient(to right, #f8e9ee, #e4bef7);
        
    }
    .st-emotion-cache-tn0cau{
        color: black;
    }
    
   .stSidebar {
    background-image: linear-gradient(to right, #4169E1 ,#f8e9ee);
    /* background-color:#4169E1;*/
    opacity: 0.9; 
    }  
</style>
    """,
    unsafe_allow_html=True
)



# ... rest of your application code ...


# ... rest of your code (st.header, st.sidebar, menu logic, etc.)

st.header("🚘Car Prediction Analysis 📈")


data = pd.read_csv('car_streamlit/Car details v3 (1).csv' ,encoding= 'latin 1')
df = pd.DataFrame(data)

st.sidebar.title("🚗 Car Analytics App")
menu = st.sidebar.radio("Go to", ["Dashboard", "Price Prediction","data_table"])


profit_percent,last_10_years_sales_change,sales_by_year_percentage_change = yearly_profit()
df1 = prepare_data(sales_by_year_percentage_change) 


def slicer():
    
    st.sidebar.header('Filter by Year')

    # 1. Get unique, sorted years
    year_options = sorted(df['year'].unique(), reverse=True)

    # 2. Create the MULTISELECT widget
    selected_years = st.sidebar.multiselect(
        'Select year(s):',
        options=year_options,
        default=year_options # Select all years by default
    )

    # 3. Filter the DataFrame
    if not selected_years:
        st.warning("Please select at least one year to view the analysis.")
        filtered_df = df.head(0) # Show empty dataframe
    else:
        # Use .isin() to filter by the list of selected years
        filtered_df = df[df['year'].isin(selected_years)].copy()
   
    return filtered_df,selected_years

def chart12(filtered_df):
    col1, col2 = st.columns(2)

    # Get and display Chart 1 in the first column
    with col1:
        # st.markdown("### Component from `get_chart_1()`")
        bar_chart = show_chart(df1)
        st.altair_chart(bar_chart, use_container_width=True)

    # Get and display Chart 2 in the second column
    with col2:
        # st.markdown("### Component from `get_chart_2()`")
        bar_chart2 = visual(filtered_df)
        st.altair_chart(bar_chart2, use_container_width=True)
        
    return col1,col2


if menu == "Dashboard":
    header()
    filtered_df,selected_years = slicer()
    chart12(filtered_df)
    chart3(df, selected_years)
    generate_transmission_chart(df, selected_years)
elif menu == "Price Prediction":
    predict()
elif menu == "data_table":
    show_table(df1)
    show_tb2()
   
    

st.sidebar.header("⚙️ prediction")
st.sidebar.subheader("🔮 Predict Future Car Price")






 









