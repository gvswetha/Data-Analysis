import pandas as pd
import matplotlib.pyplot as plt 
import streamlit as st 
from car_prediction import kpi,yearly_profit
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import altair as alt
from data_table import table


data = pd.read_csv('car_streamlit/Car details v3 (1).csv' ,encoding= 'latin 1')
df = pd.DataFrame(data)



def header(): 
    st.subheader("📊 Dashboard")
    st.markdown("""    
            <style>
            .kpi-box {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.8);
                margin: 10px;
            }
            .kpi-value {
                font-size: 28px;
                font-weight: bold;
                color: #2b8a3e;
            }
            .kpi-label {
                font-size: 16px;
                color: #6c757d;
            }
            </style>
        """, unsafe_allow_html=True)
    kpi_html = """
        <div class="kpi-box" style="background-color:#f0f2f6;padding:20px;border-radius:15px;text-align:center;">
            <div class="kpi-value" style="font-size:28px;font-weight:bold;color:#2E86C1;">{value}</div>
            <div class="kpi-label" style="font-size:16px;color:#7F8C8D;">{label}</div>
        </div>
    """

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(kpi_html.format(value=total_sales1, label="Total Sales 📈"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_html.format(value=total_brand, label="total brand "), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_html.format(value=profit_percent, label="profit 💹"), unsafe_allow_html=True)
    return col1,col2,col3  
    

profit_percent,last_10_years_sales_change,sales_by_year_percentage_change = yearly_profit()
total_sales1,total_brand,sales_by_year,percentage_sales,diff_percentage,profit_per = kpi()

 # import the table function



def visual(filtered_df):

    import altair as alt
    import pandas as pd
    import streamlit as st

    # Ensure proper data types
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    
    # Create the chart
    # Assuming 'filtered_vg' is your filtered pandas DataFrame
    bar_chart2 = (
    alt.Chart(filtered_df, title="Average Selling Price by Year")
    .mark_bar(color="#4C78A8")
    .encode(
        # Group by 'year' (Ordinal data type)
        x=alt.X('year:O', title='Year of Manufacture'),
        
        # Calculate the MEAN (Average) of 'selling_price' for each year
        y=alt.Y('selling_price:Q', aggregate='mean', title='Average Selling Price (₹)'), # <-- Key Change
        
        # Ensure the tooltip shows the aggregated mean selling price
        tooltip=[
            'year', 
            alt.Tooltip('selling_price:Q', aggregate='mean', title='Avg. Selling Price (₹)') # <-- Update tooltip
        ]
    )
    .properties(width=300, height=300)
).configure_title(
        # --- THIS IS THE KEY LINE FOR CENTERING THE TITLE ---
        anchor='middle'
    ).configure_view(
        fill='transparent', 
        stroke='transparent'
        ).interactive()
    # st.altair_chart(bar_chart, use_container_width=True)
    return bar_chart2


   
        
def chart3():    
    table2 = table()
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 7)) 

    ## Bar Plot: Total Selling Price by Brand
    sns.barplot(x=table2.index, y='sum', data=table2, ax=ax[0])
    ax[0].set_xticklabels(table2.index, rotation=90)
    ax[0].set_ylabel('Total Selling Price')
    ax[0].set_title('Total Selling Price') # Added title for clarity

    # ----------------------------------------------------------------------

    ## Line Plot: Maximum Selling Price by Brand
    # st.write('Maximum Selling Price by Brand')
    sns.lineplot(x=table2.index, y='max', data=table2, ax=ax[1])
    ax[1].set_xticklabels(table2.index, rotation=90)
    ax[1].set_ylabel('Maximum Selling Price')
    ax[1].set_title('Maximum Selling Price') # Added title for clarity


    # Adjust layout to prevent labels from overlapping
    plt.tight_layout() 

    # Display the single figure containing all three plots
    st.pyplot(fig)
    

   
