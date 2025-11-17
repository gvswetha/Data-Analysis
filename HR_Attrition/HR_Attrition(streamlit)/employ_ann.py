import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import streamlit as st
import base64

data = pd.read_csv('HR_Attrition/HR_Attrition(streamlit)/HR-Employee-Attrition.csv',encoding="latin 1")
df = pd.DataFrame(data)

df = df.rename(columns={"ï»¿Age" :"Age"})


def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return None

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

def test_train(df):
    # Clear previous TensorFlow session to avoid "pop from empty list" error
    tf.keras.backend.clear_session()

    # FIX BOM in column if present
    if 'ï»¿Age' in df.columns:
        df.rename(columns={'ï»¿Age': 'Age'}, inplace=True)

    # Features and target
    X = df[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'YearsAtCompany']]
    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

    # One-hot encoding
    X = pd.get_dummies(X, columns=['Department', 'JobRole', 'OverTime'], drop_first=True)

    print("Columns:", X.columns.tolist())  # DEBUG
    print("Shape:", X.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build Sequential model
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile and train
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, verbose=0)

    # Save model, scaler, and column order
    model.save("attrition_model.keras")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "columns.pkl")

    print("Model and preprocessing saved successfully!")
    return None

    

def feature_eng():
    
   
    test_train(df)
    
   
        
    count_dept = df['Department'].value_counts()
    st.title("Department vs Employee")
    st.write(count_dept)

    job_role = df['JobRole'].value_counts()
    st.title("Job Role VS Employee")
    st.write(job_role)

    #department under job_role

    dept_job_role = df.groupby('Department')['JobRole'].value_counts()
    st.title("Department VS Employee")
    st.write(dept_job_role)

    # monthly income range under the particular dept and jobrole

    monthly_income_range = df.groupby(['Department', 'JobRole'])['MonthlyIncome'].agg(['min', 'max'])
    st.title("Department vs Salary Range")
    st.write(monthly_income_range)
    
    return monthly_income_range

def eda():
    
        def shorten_number_builtin(num):
            """
            Converts a number to a human-readable short string (e.g., 5187873253 -> '5.19B').
            This uses ONLY built-in Python features (no external libraries).
            """
            # Ensure the input is a float for accurate division
            num = float(num)
            
            # Define the magnitude thresholds and their suffixes
            # (10^0, 10^3, 10^6, 10^9, 10^12)
            suffixes = ['', 'K', 'M', 'B', 'T']
            
            # Loop through the powers of 10 in descending order
            # Starts at 10^12 (T) and works down to 10^0
            for i in range(len(suffixes) - 1, -1, -1):
                power = 10**(i * 3)
                
                # Check if the number is large enough for the current suffix
                if abs(num) >= power:
                    # Format the number: Divide, round to 2 decimal places, and append suffix
                    short_num = round(num / power, 2)
                    return f"{short_num}{suffixes[i]}"
                    
            # If the number is < 1000, return it as is
            return str(num)
        
        percentage_att_f = (df['Attrition'] == 'Yes').mean() * 100
        # print(f"Percentage of People Attrition: {percentage_att:.0f}%")

        average_age_f = df['Age'].mean()
        # print(f"Average Age: {average_age:.0f}")

        average_companyyears_f= df['YearsAtCompany'].mean()
        # print(f"Average Years at Company: {average_companyyears:.0f}")
        average_promotionyear_f = df['YearsSinceLastPromotion'].mean()
        # print(f"Average Years Since Last Promotion: {average_promotionyear:.0f}")
        
        percentage_att = int(round(percentage_att_f))
        average_age = int(round(average_age_f))
        average_companyyears = int(round(average_companyyears_f))
        average_promotionyear = int(round(average_promotionyear_f))
        
        total_employee = df['EmployeeCount'].count()
        print(total_employee)
        
        avg_overtime = df.groupby('OverTime')['MonthlyIncome'].mean()
        print(avg_overtime)

        rel_stastifaction = df.groupby('JobRole')['JobSatisfaction'].mean()
        print(rel_stastifaction)
        
        return percentage_att,average_age,average_companyyears,total_employee

def kpi():
    percentage_att,average_age,average_companyyears,total_employee = eda()
    kpi_html = """
        <div class="glass-effect">
            <div class="kpi-box" style="background-color:#f0f2f6;padding:20px;border-radius:15px;text-align:center;">
            <div class="kpi-value" style="font-size:28px;font-weight:bold;color:#2E86C1;">{value}</div>
            <div class="kpi-label" style="font-size:16px;color:#7F8C8D;">{label}</div>
       </div></div>
    """
    
    col1, col2, col3 ,col4 = st.columns(4)

    with col1:
        st.markdown(kpi_html.format(value=percentage_att, label="Attrition % 📈"), unsafe_allow_html=True)
    with col2:
        
        st.markdown(kpi_html.format(value=average_age, label="AVG AGE"), unsafe_allow_html=True)

    with col3:
        
        st.markdown(kpi_html.format(value=total_employee, label="Employee 👨🏻‍💼"), unsafe_allow_html=True)

    with col4:
        
         st.markdown(kpi_html.format(value=average_companyyears, label="AVG year"), unsafe_allow_html=True)
             
    return col1,col2,col3,col4
    
    
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import shap


def predict():
    
    # Clear TF session to prevent errors
    tf.keras.backend.clear_session()

    # Load model, scaler, and column names
    model = tf.keras.models.load_model("attrition_model.keras")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("columns.pkl")

    # ------------------------------ UI ------------------------------
    st.subheader("Employee Attrition Prediction Dashboard 🧑‍💼📊")
    st.write("Enter employee details below to predict attrition probability.")

    # USER INPUTS
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    jobrole = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative",
        "Manager", "Sales Representative", "Research Director", "Human Resources"
    ])
    income = st.number_input("Monthly Income", min_value=1000, max_value=80000, value=5000)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    years = st.number_input("Years at Company", min_value=0, max_value=40, value=5)

    if st.button("Predict"):
        # ------------------------ PREPROCESS ------------------------
        user_df = pd.DataFrame([{
            "ï»¿Age": age,
            "Department": department,
            "JobRole": jobrole,
            "MonthlyIncome": income,
            "OverTime": overtime,
            "YearsAtCompany": years
        }])

        # One-hot encode and match model columns
        user_df = pd.get_dummies(user_df, columns=["Department", "JobRole", "OverTime"], drop_first=True)
        user_df = user_df.reindex(columns=model_columns, fill_value=0)

        # Scale
        user_scaled = scaler.transform(user_df)

        # ------------------------ MODEL PREDICTION ------------------------
        probability = float(model.predict(user_scaled)[0][0])

        # ------------------------ RESULT ------------------------
        st.subheader("🔍 Prediction Result")
        if probability > 0.5:
            st.error(f"🚨 High Attrition Risk: **{probability:.2f}**")
        else:
            st.success(f"✅ Low Attrition Risk: **{probability:.2f}**")

        # ------------------------ GAUGE ------------------------
        st.subheader("📌 Attrition Probability Gauge")
        fig, ax = plt.subplots(figsize=(6,2))
        ax.barh(["Attrition Probability"], [probability], color="skyblue")
        ax.set_xlim([0,1])
        st.pyplot(fig)

        # ------------------------ SHAP EXPLAINABILITY ------------------------
        st.subheader("🌟 Feature Contribution (Local SHAP)")

        # Use KernelExplainer for small input
        explainer = shap.KernelExplainer(model.predict, np.zeros((1, len(model_columns))))
        shap_values = explainer.shap_values(user_scaled, nsamples=100)

        # Waterfall plot for one row
        shap.plots._waterfall.waterfall_legacy(
            expected_value=explainer.expected_value[0],
            shap_values=shap_values[0][0],
            feature_names=model_columns,
            max_display=10,
            show=True
        )

        # ------------------------ GLOBAL FEATURE IMPORTANCE ------------------------
        st.subheader("🔥 Feature Importance (Global)")

        shap_global = np.mean(np.abs(shap_values[0]), axis=0)
        indices = np.argsort(shap_global)[-10:][::-1]
        importances = shap_global[indices]
        labels = np.array(model_columns)[indices]

        fig3, ax3 = plt.subplots(figsize=(6,5))
        ax3.barh(labels, importances, color="salmon")
        ax3.set_title("Top 10 Important Features")
        ax3.invert_yaxis()
        st.pyplot(fig3)
        
        return None
    
import plotly.express as px
import plotly.graph_objects as go
   
def charts():

    # -----------------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------------
    data = pd.read_csv('HR_Attrition/HR_Attrition(streamlit)/HR-Employee-Attrition.csv', encoding="latin 1")
    df = pd.DataFrame(data)
    df = df.rename(columns={"ï»¿Age": "Age"})

    # -----------------------------------------------------------
    # STYLE FUNCTION (APPLIED TO ALL CHARTS)
    # -----------------------------------------------------------
    def style_chart(fig):
        fig.update_layout(
            plot_bgcolor="rgba(255,255,255,0.3)",   # semi-transparent white
            paper_bgcolor="rgba(255,255,255,0.3)",  # semi-transparent white

            # axis styling (white background theme → black text)
            xaxis=dict(
                title=dict(font=dict(color="black")),
                tickfont=dict(color="black"),
                showgrid=False
            ),
            yaxis=dict(
                title=dict(font=dict(color="black")),
                tickfont=dict(color="black"),
                showgrid=False
            ),

            # general font color
            font=dict(color="black"),

            # hover box style
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.8)",  
                font_color="black",
                bordercolor="black"
            )
        )
        return fig

    # -----------------------------------------------------------
    # SIDEBAR FILTERS
    # -----------------------------------------------------------
    st.sidebar.header("Filters")
    department_filter = st.sidebar.multiselect(
        "Department", df["Department"].unique(), df["Department"].unique()
    )
    job_filter = st.sidebar.multiselect(
        "Job Role", df["JobRole"].unique(), df["JobRole"].unique()
    )

    df = df[
        (df["Department"].isin(department_filter)) &
        (df["JobRole"].isin(job_filter))
    ]

    # -----------------------------------------------------------
    # SALARY CATEGORY TABLE
    # -----------------------------------------------------------
    st.subheader("Salary Category Table")

    salary_map = {
        "Low Income": df[df["MonthlyIncome"] < 4000],
        "Medium Income": df[(df["MonthlyIncome"] >= 4000) & (df["MonthlyIncome"] < 9000)],
        "High Income": df[df["MonthlyIncome"] >= 9000],
    }

    table_data = []
    for category, data_group in salary_map.items():
        table_data.append({
            "Salary Category": category,
            "% Attrition": round((data_group["Attrition"] == "Yes").mean() * 100, 1),
            "Employee Count": len(data_group),
            "Avg. Performance Rating": round(data_group["PerformanceRating"].mean(), 1),
            "Max Years at Company": data_group["YearsAtCompany"].max()
        })

    st.dataframe(pd.DataFrame(table_data))

    # -----------------------------------------------------------
    # BAR + LINE COMBO
    # -----------------------------------------------------------
    st.subheader("Total Working Years (Group)")

    group = df.groupby("TotalWorkingYears").size().reset_index(name="Count")
    fig_bar_line = go.Figure()

    fig_bar_line.add_trace(go.Bar(
        x=group["TotalWorkingYears"],
        y=group["Count"],
        name="Employee Count"
    ))

    fig_bar_line.add_trace(go.Scatter(
        x=group["TotalWorkingYears"],
        y=group["Count"],
        mode="lines+markers+text",
        text=group["Count"],
        textposition="top center",
        name="Trend"
    ))

    fig_bar_line = style_chart(fig_bar_line)
    st.plotly_chart(fig_bar_line, use_container_width=True, key="barline")

    # -----------------------------------------------------------
    # ATTRITION / MARITAL STATUS
    # -----------------------------------------------------------
    st.subheader("Attrition / Marital Status")

    group = df.groupby(["Attrition", "MaritalStatus"]).size().reset_index(name="Count")

    fig_attri = px.bar(
        group,
        x="MaritalStatus",
        y="Count",
        color="Attrition",
        barmode="group",
        text="Count"
    )

    fig_attri.update_traces(textposition="auto")
    fig_attri = style_chart(fig_attri)

    st.plotly_chart(fig_attri, use_container_width=True, key="attri_marital")

    # -----------------------------------------------------------
    # DONUT CHARTS
    # -----------------------------------------------------------
    def donut_chart(data, label, title):
        fig = px.pie(data, values="Count", names=label, hole=0.5)
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(title=title)
        return style_chart(fig)

    # donut1 – Divorce vs Attrition
    divorce_data = df.groupby("MaritalStatus").size().reset_index(name="Count")
    donut1 = donut_chart(divorce_data, "MaritalStatus", "Divorce vs Attrition")

    # donut2 – Age groups
    age_bins = [18, 25, 35, 45, 60]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins).astype(str)
    age_data = df.groupby("AgeGroup").size().reset_index(name="Count")

    donut2 = px.pie(age_data, values="Count", names="AgeGroup", hole=0.5)
    donut2.update_traces(textinfo="percent+label", textfont_size=14)
    donut2 = style_chart(donut2)
    

    # donut3 – Years at Company
    yearatcompany_bins = [0,5,10,15,25,35,40]
    df['YearGroup'] = pd.cut(df["YearsAtCompany"], bins = yearatcompany_bins).astype(str)
    year_data = df.groupby("YearGroup").size().reset_index(name="Count")
    donut3 = donut_chart(year_data, "YearGroup", "Years at Company vs Attrition")

    # donut4 – Salary Category vs Count
    salary_data = pd.DataFrame(table_data)
    donut4 = px.pie(salary_data, values="Employee Count", names="Salary Category", hole=0.5)
    donut4.update_traces(textinfo="percent+label")
    
    donut4 = style_chart(donut4)

    # -----------------------------------------------------------
    # DISPLAY DONUTS IN 2×2 GRID
    # -----------------------------------------------------------
    c11, c22 = st.columns(2)
    with c11:
        st.subheader("Divorce vs Attrition")
        st.plotly_chart(donut1, use_container_width=True, key="donut1")
    with c22:
        st.subheader("Age Group vs Attrition")
        st.plotly_chart(donut2, use_container_width=True, key="donut2")

    c33, c44 = st.columns(2)
    with c33:
        st.subheader("Years at Company vs Attrition")
        st.plotly_chart(donut3, use_container_width=True, key="donut3")
    with c44:
        st.subheader("Salary vs employee") 
        st.plotly_chart(donut4, use_container_width=True, key="donut4")

    return None

