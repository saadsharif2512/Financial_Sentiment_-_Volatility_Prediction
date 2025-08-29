import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Financial Sentiment & Volatility Prediction",
    layout="wide"
)

# Caching Functions for Performance 
@st.cache_data
def load_data():
    """Loads the pre-processed data from CSV files."""
    try:
        daily_df = pd.read_csv('final_daily_data.csv', parse_dates=['date'])
        all_news_df = pd.read_csv('all_news_data.csv', parse_dates=['scraped_at'])
        return daily_df, all_news_df
    except FileNotFoundError:
        st.error("Data files not found. Please run the `analysis.ipynb` notebook to generate them.")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_resource
def load_model():
    """Loads the trained XGBoost model from the file."""
    try:
        model = joblib.load('xgboost_volatility_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file ('xgboost_volatility_model.joblib') not found. Please run the notebook to train and save the model.")
        return None

daily_df, all_news_df = load_data()
model = load_model()

# Main App UI 
st.title("Financial Sentiment & Volatility Prediction")

if not daily_df.empty and model is not None:
    st.sidebar.title("Controls")
    tickers = sorted(daily_df['ticker'].unique().tolist())
    selected_ticker = st.sidebar.selectbox('Select a Stock Ticker', tickers)

    # Filter data for the selected ticker
    ticker_daily_df = daily_df[daily_df['ticker'] == selected_ticker].copy()
    ticker_news_df = all_news_df[all_news_df['ticker'] == selected_ticker].copy()

    st.header(f"Dashboard for {selected_ticker}")

    # Display Latest Prediction
    st.subheader("Latest Volatility Forecast")
    latest_data = ticker_daily_df.sort_values('date').tail(1).copy()

    if not latest_data.empty:
        feature_cols = [col for col in model.feature_names_in_]
        features_for_prediction = latest_data[feature_cols]
        
        prediction = model.predict(features_for_prediction)
        latest_actual_volatility = latest_data['volatility'].values[0]
        
        st.metric(
            label="Predicted Volatility for the Next Trading Day",
            value=f"{prediction[0]:.6f}",
            delta=f"{(prediction[0] - latest_actual_volatility):.6f} vs. Last Actual",
            delta_color="off"
        )
    else:
        st.warning("Not enough recent data to make a new prediction.")

    # Visualizations
    st.header("Historical Data Visualization")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stock Closing Price")
        st.line_chart(ticker_daily_df.set_index('date')['Close'])
        st.subheader("Daily News Volume")
        st.bar_chart(ticker_daily_df.set_index('date')['news_volume'])
    with col2:
        st.subheader("Average Daily News Sentiment")
        st.line_chart(ticker_daily_df.set_index('date')['mean_sentiment'])
        st.subheader("Volatility (Risk)")
        st.line_chart(ticker_daily_df.set_index('date')['volatility'])

    # Recent News Articles
    st.header("Recent News Articles")
    recent_news = ticker_news_df.sort_values('scraped_at', ascending=False).head(5)
    for index, row in recent_news.iterrows():
        st.markdown(f"**{row['headline']}**")
        st.caption(f"Scraped on: {pd.to_datetime(row['scraped_at']).strftime('%Y-%m-%d %H:%M')}")
        st.caption(f"link: {row['link']}")
        if 'sentiment_score' in row and pd.notna(row['sentiment_score']):
            st.markdown(f"Sentiment Score: **{row['sentiment_score']:.2f}**")
        if 'summary' in row and pd.notna(row['summary']):
            st.markdown(f"> {row['summary']}")
        st.markdown("---")