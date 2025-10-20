import streamlit as st
import pandas as pd
import altair as alt
from car_prediction import pyfunction



def prepare_data(sales_by_year_percentage_change):
    """
    Prepares a DataFrame with numeric and formatted values for table/chart
    """
    table_data = []
    for year, selling_price in sales_by_year_percentage_change.items():
        status = "Profit ⬆️" if selling_price > 0 else "Loss ⬇️" if selling_price < 0 else "No Change ↔️"
        table_data.append({
            "year": year,
            "selling_price": selling_price,        # numeric for chart
            "selling_price_str": f"{selling_price:+.2f}%",  # formatted string for table
            "Status": status
        })
    df1 = pd.DataFrame(table_data)
    
    return df1

def color_status_rows(s):
    """
    Returns a CSS style string for the 'selling_price_str' column based on the 'Status'.
    This will highlight the cell with the percentage change.
    """
    if s['Status'].startswith('Profit'):
        return ['background-color: #d4edda; color: #155724'] * len(s) # Light Green background
    elif s['Status'].startswith('Loss'):
        return ['background-color: #f8d7da; color: #721c24'] * len(s) # Light Red background
    else:
        return ['background-color: #f8f9fa; color: #343a40'] * len(s)
    

def show_table(df1):
    """
    Displays the formatted table
    """
    
    styled_df1 = df1.style.apply(
        color_status_rows, 
        axis=1, # Apply row-wise
    )
    
    # Display the styled dataframe
    st.dataframe(styled_df1, use_container_width=True)
    # st.subheader("Yearly Sales Change Table")
    # st.dataframe(df1[["Year", "selling_price_str", "Status"]])

def show_chart(df1):
    """
    Displays the colored bar chart 
    """
    # st.subheader("Yearly Sales Change Chart")
   # Assuming 'filtered_data' is the name of your filtered DataFrame
    # Use the actual variable name for your filtered data
    bar_chart = alt.Chart(df1).mark_bar().encode( # <-- Change df1 to the filtered data variable
        x=alt.X("year:O", title="Year"),
        y=alt.Y("selling_price:Q", title="Sales Change (%)"),
        color=alt.Color(
            "Status:N",
            scale=alt.Scale(
                domain=["Profit ⬆️", "Loss ⬇️", "No Change ↔️"],
                range=["green", "red", "gray"]
            )
        ),
        tooltip=["year", "selling_price", "Status"]
    ).properties(
        width=300,
        height=300,
        title="Yearly Sales Percentage Change"
    ).configure_title(
        # --- THIS IS THE KEY LINE FOR CENTERING THE TITLE ---
        anchor='middle'
    ).configure_view(
        fill='transparent', 
        stroke='transparent'
        ).interactive()
    
    return bar_chart
    
    
    
def table():
    df = pyfunction()
    brand_selling = df.groupby('brand')['selling_price'].agg(['min', 'max', 'sum'])
    print("\nBrand-wise Total Selling Price:")
    print(brand_selling)

    table2 = pd.DataFrame(brand_selling)
    
    return table2

def show_tb2():
    table2 = table()
    st.dataframe(table2)
    
def chart3(df, selected_years):
    df = pyfunction()
    dc = df.groupby(['brand', 'year'])['selling_price'].count().rename('Count')
    df_chart = dc.reset_index()

    # 2. TYPE CASTING (Essential for Altair)
    df_chart['brand'] = df_chart['brand'].astype(str)
    # The year column must be string (Ordinal :O) for a clean discrete Altair axis
    df_chart['year'] = df_chart['year'].astype(str) 
    
    # 3. APPLY THE FILTER (This is the missing step)
    # Filter the aggregated and prepared data based on the years selected elsewhere
    if selected_years:
        filtered_df_for_chart = df_chart[df_chart['year'].isin(map(str, selected_years))].copy()
    else:
        # If no years are selected, treat it as empty
        filtered_df_for_chart = df_chart.head(0)

    # st.subheader("Count of Listings by Brand and Year")
      
    if not filtered_df_for_chart.empty:
        # 4. CHART CREATION (Use the correctly filtered and prepared DataFrame)
        line_chart = alt.Chart(filtered_df_for_chart).mark_line(point=True).encode(
            x=alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Count:Q', title='Number of Listings'),
            color=alt.Color('brand:N', title='Brand'),
            tooltip=['year', 'brand', 'Count']
        ).properties(
            title={"text": "Yearly Listing Count by Brand", "anchor": "middle"}
        ).configure_view(
            fill='transparent', 
            stroke='transparent'
        ).interactive()

        st.altair_chart(line_chart, use_container_width=True)
        return line_chart, dc
    else:
        st.info("No data to display for the selected years.")
        return None, dc
  
import streamlit as st
import pandas as pd
import altair as alt

def generate_transmission_chart(df, selected_years):
    """
    Groups raw data by year and transmission, counts selling price, filters by 
    selected_years, and generates an Altair line chart.
    """
    st.subheader("Yearly Listing Count by Transmission Type")

    # 1. AGGREGATE THE DATA
    # dc is a Pandas Series with MultiIndex (year, transmission)
    dc = df.groupby(['year', 'transmission'])['selling_price'].count().rename('Count')
    
    # Convert MultiIndex Series to a standard DataFrame for Altair
    df_chart = dc.reset_index()

    # 2. TYPE CASTING (Essential for Altair)
    # Ensure columns are explicit string types for clean encoding
    df_chart['transmission'] = df_chart['transmission'].astype(str)
    df_chart['year'] = df_chart['year'].astype(str) 
    
    # 3. APPLY THE FILTER
    if selected_years:
        # Map selected_years (likely integers) to string for filtering the 'year' column
        selected_years_str = list(map(str, selected_years))
        filtered_df_for_chart = df_chart[df_chart['year'].isin(selected_years_str)].copy()
    else:
        st.info("No years selected. Please use the sidebar filter.")
        return

    # 4. CHART CREATION
    if not filtered_df_for_chart.empty:
        line_chart = alt.Chart(filtered_df_for_chart).mark_line(point=True).encode(
            # X-axis: Year (Ordinal 'O')
            x=alt.X('year:O', title='Year', axis=alt.Axis(labelAngle=0)),
            # Y-axis: Count (Quantitative 'Q')
            y=alt.Y('Count:Q', title='Number of Listings'),
            # Color: Transmission (Nominal 'N')
            color=alt.Color('transmission:N', title='Transmission'),
            tooltip=['year', 'transmission', 'Count']
        ).properties(
            title={
                "text": "Yearly Listing Count by Transmission Type",
                "anchor": "middle"
            }
        ).configure_view(
            fill='transparent', 
            stroke='transparent'
        ).interactive()

        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.info("No data to display for the selected filter.") 
        
  