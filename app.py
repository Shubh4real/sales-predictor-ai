import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import traceback
import google.generativeai as genai
import os
from model import (
    clean_data, handle_outliers,
    engineer_features, scale_features,
    FEATURES, TARGET
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Sales Predictor AI",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.title("📈 Sales Predictor AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Explore Data", "🔮 Predict Sales", "🤖 AI Explanation", "📂 Batch Prediction"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Python, scikit-learn & Gemini AI")

# ─────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────
if page == "🏠 Home":
    st.title("📈 Sales Predictor AI")
    st.markdown("### AI-powered sales forecasting with natural language insights")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 📊 Explore\nUpload and visualise your sales data with interactive charts")
    with col2:
        st.success("### 🔮 Predict\nRun a trained machine learning model to forecast sales")
    with col3:
        st.warning("### 🤖 Explain\nGet plain-English AI explanations of your predictions")

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown("""
    1. Upload your sales CSV file or use the built-in dataset
    2. Explore trends, patterns and distributions in your data
    3. Run the prediction model to forecast future sales
    4. Get AI-generated insights explaining what the predictions mean
    5. Run batch predictions on an entire test dataset at once
    """)

    st.markdown("---")
    st.markdown("#### Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R2 Score",      "0.923")
    col2.metric("MAE",           "1,086 units")
    col3.metric("Training Rows", "66,766")
    col4.metric("CV Std",        "0.0023")

# ─────────────────────────────────────────
# PAGE 2 — EXPLORE DATA
# ─────────────────────────────────────────
elif page == "📊 Explore Data":
    st.title("📊 Explore Data")
    st.markdown("---")

    uploaded = st.file_uploader("Upload your train.csv file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=['date'])
    else:
        st.info("No file uploaded — using built-in dataset")
        df = pd.read_csv('data/train.csv', parse_dates=['date'])

    if 'sales' not in df.columns:
        st.error(f"❌ Wrong file! Expected a 'sales' column but found: {df.columns.tolist()}")
        st.info("Please upload **train.csv** not test.csv — the test file has no sales data.")
        st.stop()

    st.success(f"✓ Loaded {len(df):,} rows")

    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows",       f"{len(df):,}")
    col2.metric("Stores",           f"{df['store_nbr'].nunique()}")
    col3.metric("Product Families", f"{df['family'].nunique()}")
    col4.metric("Date Range",       f"{df['date'].min().date()} → {df['date'].max().date()}")

    st.markdown("---")

    st.subheader("Daily Sales Trend")
    daily = df.groupby('date')['sales'].sum().reset_index()
    fig1  = px.line(daily, x='date', y='sales',
                    title='Total Daily Sales Over Time',
                    color_discrete_sequence=['#1f77b4'])
    fig1.update_layout(xaxis_title='Date', yaxis_title='Total Sales')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Sales by Store")
    store_sales = df.groupby('store_nbr')['sales'].sum().reset_index()
    fig2 = px.bar(store_sales, x='store_nbr', y='sales',
                  title='Total Sales by Store',
                  color='sales', color_continuous_scale='Blues')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Product Family")
    family_sales = (df.groupby('family')['sales']
                    .sum().reset_index()
                    .sort_values('sales', ascending=True))
    fig3 = px.bar(family_sales, x='sales', y='family',
                  orientation='h', title='Total Sales by Product Family',
                  color='sales', color_continuous_scale='Greens')
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    df['month_year'] = df['date'].dt.to_period('M').astype(str)
    monthly = df.groupby('month_year')['sales'].sum().reset_index()
    fig4 = px.bar(monthly, x='month_year', y='sales',
                  title='Monthly Total Sales',
                  color='sales', color_continuous_scale='Oranges')
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Sales Distribution")
    daily_store = df.groupby(['date', 'store_nbr'])['sales'].sum().reset_index()
    fig5 = px.histogram(daily_store, x='sales', nbins=50,
                        title='Distribution of Daily Sales per Store',
                        color_discrete_sequence=['#2ca02c'])
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Promotion Impact on Sales")
    promo = df.groupby('onpromotion')['sales'].mean().reset_index().head(20)
    fig6  = px.scatter(promo, x='onpromotion', y='sales',
                       title='Average Sales vs Items on Promotion',
                       trendline='ols')
    st.plotly_chart(fig6, use_container_width=True)

    with st.expander("View Raw Data"):
        st.dataframe(df.head(100))

# ─────────────────────────────────────────
# PAGE 3 — PREDICT
# ─────────────────────────────────────────
elif page == "🔮 Predict Sales":
    st.title("🔮 Predict Sales")
    st.markdown("---")

    try:
        model, scaler = load_model()
        st.success("✓ Model loaded successfully")
    except Exception as e:
        st.error(f"Model not found: {e}. Please run model.py first.")
        st.stop()

    st.markdown("### Select prediction parameters")

    col1, col2 = st.columns(2)
    with col1:
        store  = st.selectbox("Store Number", list(range(1, 55)))
        month  = st.selectbox("Month", list(range(1, 13)),
                               format_func=lambda x: pd.Timestamp(2017, x, 1).strftime('%B'))
        day    = st.slider("Day of Month", 1, 28, 15)
        year   = st.selectbox("Year", [2015, 2016, 2017, 2018])
    with col2:
        onpromotion = st.slider("Avg Items on Promotion", 0.0, 10.0, 2.0)
        lag_7_val   = st.number_input("Sales 7 days ago",  value=10000.0, step=500.0)
        lag_30_val  = st.number_input("Sales 30 days ago", value=9500.0,  step=500.0)

    date         = pd.Timestamp(year, month, day)
    day_of_week  = date.dayofweek
    quarter      = date.quarter
    week_of_year = date.isocalendar()[1]
    is_weekend   = int(day_of_week >= 5)
    rolling_7    = lag_7_val * 0.95

    input_data = pd.DataFrame([{
        'store_nbr':       store,
        'day_of_week':     day_of_week,
        'month':           month,
        'year':            year,
        'day':             day,
        'is_weekend':      is_weekend,
        'quarter':         quarter,
        'week_of_year':    week_of_year,
        'avg_onpromotion': onpromotion,
        'lag_7':           lag_7_val,
        'lag_30':          lag_30_val,
        'rolling_7':       rolling_7
    }])

    st.markdown("---")

    if st.button("🔮 Run Prediction", type="primary"):
        input_scaled = scaler.transform(input_data[FEATURES])
        prediction   = model.predict(input_scaled)[0]

        st.session_state['prediction'] = prediction
        st.session_state['input_data'] = input_data
        st.session_state['store']      = store
        st.session_state['date']       = date

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Sales", f"{prediction:,.0f} units")
        col2.metric("Store",           f"Store {store}")
        col3.metric("Date",            str(date.date()))

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prediction,
            delta = {'reference': lag_7_val},
            title = {'text': "Predicted Sales vs Last Week"},
            gauge = {
                'axis': {'range': [0, 30000]},
                'bar':  {'color': "#1f77b4"},
                'steps': [
                    {'range': [0,     10000], 'color': '#f0f0f0'},
                    {'range': [10000, 20000], 'color': '#d0e4f7'},
                    {'range': [20000, 30000], 'color': '#a0c8f0'},
                ],
                'threshold': {
                    'line':      {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value':     lag_7_val
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        compare = pd.DataFrame({
            'Period': ['30 days ago', '7 days ago', 'Predicted'],
            'Sales':  [lag_30_val, lag_7_val, prediction]
        })
        fig2 = px.bar(compare, x='Period', y='Sales',
                      title='Sales Comparison', color='Period',
                      color_discrete_sequence=['#aec7e8', '#1f77b4', '#ff7f0e'])
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Input Summary")
        st.dataframe(input_data)
        st.info("👉 Go to **AI Explanation** page for plain-English insights!")

# ─────────────────────────────────────────
# PAGE 4 — AI EXPLANATION
# ─────────────────────────────────────────
elif page == "🤖 AI Explanation":
    st.title("🤖 AI Explanation")
    st.markdown("---")

    st.markdown("""
    Get a plain-English explanation of your sales prediction powered by **Google Gemini AI**.
    👉 Get your free API key at [aistudio.google.com](https://aistudio.google.com)
    """)

    api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Free at https://aistudio.google.com — no credit card needed"
    )

    if 'prediction' not in st.session_state:
        st.warning("⚠ Please run a prediction first on the Predict Sales page.")
        st.stop()

    prediction = st.session_state['prediction']
    store      = st.session_state['store']
    date       = st.session_state['date']
    input_data = st.session_state['input_data']

    st.markdown("---")
    st.subheader("Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Store",           f"Store {store}")
    col2.metric("Date",            str(date.date()))
    col3.metric("Predicted Sales", f"{prediction:,.0f} units")
    st.markdown("---")

    if st.button("🤖 Generate AI Explanation", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key above.")
            st.stop()

        try:
            os.environ["GOOGLE_API_KEY"] = api_key
            client = genai.GenieClient()
            response = client.generate_text(
                model="gemini-2.0-flash",
                prompt=prompt
            )
            explanation = getattr(response, "text", None)
            if explanation is None:
                explanation = response.output[0].content[0].text

            st.success("✓ AI Explanation Generated")
            st.markdown("### 📋 Business Insight Report")
            st.markdown(explanation)
            st.markdown("---")

            st.download_button(
                label     = "📥 Download Report",
                data      = explanation,
                file_name = f"sales_report_store{store}_{date.date()}.txt",
                mime      = "text/plain"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())

# ─────────────────────────────────────────
# PAGE 5 — BATCH PREDICTION
# ─────────────────────────────────────────
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("---")

    st.markdown("""
    Upload your **test.csv** file to generate sales predictions for all rows at once.
    The app will automatically engineer features and run the model on every row.
    """)

    with st.expander("ℹ️ What file format is expected?"):
        st.markdown("""
        Your CSV needs these columns:
        - `id` — row identifier
        - `date` — date in YYYY-MM-DD format
        - `store_nbr` — store number (1–54)
        - `family` — product family name
        - `onpromotion` — number of items on promotion
        """)

    uploaded_test = st.file_uploader(
        "Upload test.csv for batch predictions", type=["csv"]
    )

    st.write("File detected:", uploaded_test is not None)

    if uploaded_test is not None:
        st.info("File uploaded! Loading model and running predictions...")

        try:
            # Load model
            model, scaler = load_model()
            st.success("✓ Model loaded")

            # Load CSV
            test_df = pd.read_csv(uploaded_test, parse_dates=['date'])
            st.success(f"✓ Loaded {len(test_df):,} rows")
            st.write("Columns found:", test_df.columns.tolist())

            # Validate
            required_cols = ['date', 'store_nbr', 'onpromotion']
            missing_cols  = [c for c in required_cols if c not in test_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                st.stop()

            st.subheader("Preview of uploaded file")
            st.dataframe(test_df.head(10))
            st.markdown("---")

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Running predictions on all rows..."):

                    df_pred = test_df.copy().sort_values(['store_nbr', 'date'])

                    # Feature engineering
                    df_pred['day_of_week']     = df_pred['date'].dt.dayofweek
                    df_pred['month']           = df_pred['date'].dt.month
                    df_pred['year']            = df_pred['date'].dt.year
                    df_pred['day']             = df_pred['date'].dt.day
                    df_pred['quarter']         = df_pred['date'].dt.quarter
                    df_pred['week_of_year']    = df_pred['date'].dt.isocalendar().week.astype(int)
                    df_pred['is_weekend']      = (df_pred['day_of_week'] >= 5).astype(int)
                    df_pred['avg_onpromotion'] = df_pred['onpromotion']

                    # Lag estimates
                    df_pred['lag_7']     = df_pred.groupby('store_nbr')['onpromotion'].transform('mean') * 1000 + 9000
                    df_pred['lag_30']    = df_pred['lag_7'] * 0.95
                    df_pred['rolling_7'] = df_pred['lag_7'] * 0.97

                    # Scale and predict
                    X        = df_pred[FEATURES]
                    X_scaled = scaler.transform(X)
                    preds    = model.predict(X_scaled)

                    df_pred['predicted_sales'] = np.round(preds, 2)
                    df_pred['predicted_sales'] = df_pred['predicted_sales'].clip(lower=0)

                    st.success(f"✓ Predictions generated for {len(df_pred):,} rows!")
                    st.markdown("---")

                    # Summary metrics
                    st.subheader("Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Rows",          f"{len(df_pred):,}")
                    col2.metric("Avg Predicted Sales", f"{df_pred['predicted_sales'].mean():,.0f}")
                    col3.metric("Max Predicted Sales", f"{df_pred['predicted_sales'].max():,.0f}")
                    col4.metric("Min Predicted Sales", f"{df_pred['predicted_sales'].min():,.0f}")

                    st.markdown("---")

                    # Chart — by store
                    st.subheader("Predicted Sales by Store")
                    store_preds = df_pred.groupby('store_nbr')['predicted_sales'].sum().reset_index()
                    fig1 = px.bar(store_preds, x='store_nbr', y='predicted_sales',
                                  title='Total Predicted Sales by Store',
                                  color='predicted_sales',
                                  color_continuous_scale='Blues')
                    st.plotly_chart(fig1, use_container_width=True)

                    # Chart — over time
                    st.subheader("Predicted Sales Over Time")
                    daily_preds = df_pred.groupby('date')['predicted_sales'].sum().reset_index()
                    fig2 = px.line(daily_preds, x='date', y='predicted_sales',
                                   title='Total Daily Predicted Sales',
                                   color_discrete_sequence=['#ff7f0e'])
                    st.plotly_chart(fig2, use_container_width=True)

                    # Chart — by day of week
                    st.subheader("Predicted Sales by Day of Week")
                    dow_map     = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
                    df_pred['day_name'] = df_pred['day_of_week'].map(dow_map)
                    dow_preds   = df_pred.groupby('day_name')['predicted_sales'].mean().reset_index()
                    fig3 = px.bar(dow_preds, x='day_name', y='predicted_sales',
                                  title='Average Predicted Sales by Day of Week',
                                  color='predicted_sales',
                                  color_continuous_scale='Greens',
                                  category_orders={'day_name': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']})
                    st.plotly_chart(fig3, use_container_width=True)

                    # Results table
                    st.subheader("Full Prediction Results")
                    results = df_pred[['date', 'store_nbr', 'predicted_sales']].copy()
                    if 'id' in test_df.columns:
                        results.insert(0, 'id', test_df['id'].values[:len(results)])
                    st.dataframe(results)

                    # Download
                    st.download_button(
                        label     = "📥 Download Predictions CSV",
                        data      = results.to_csv(index=False),
                        file_name = "batch_predictions.csv",
                        mime      = "text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(traceback.format_exc())

    else:
        st.info("👆 Upload your test.csv file above to get started.")