import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import gspread
from gspread_dataframe import get_as_dataframe
import warnings

# Suppress ignorable warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- App Configuration ---
st.set_page_config(
    page_title="Financial Sentiment & Volatility Prediction",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_data(ttl="1h") # Cache the data for 1 hour
def load_and_combine_data():
    """
    This function loads the historical data from CSV and the live data from Google Sheets,
    then combines and processes them.
    """
    # --- Load Historical Data from CSV ---
    try:
        historical_df = pd.read_csv('historical_news_data.csv')
        historical_df.columns = historical_df.columns.str.lower().str.strip()
    except FileNotFoundError:
        historical_df = pd.DataFrame()

    # --- Load Live Data from Google Sheets ---
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        worksheet = gc.open("Financial News Scraper").sheet1
        live_df = get_as_dataframe(worksheet, header=0)
        live_df.columns = live_df.columns.str.lower().str.strip()
    except Exception as e:
        st.error(f"Could not load live data from Google Sheet. Error: {e}")
        live_df = pd.DataFrame()

    # --- Combine and Process ---
    df = pd.concat([historical_df, live_df], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates(subset=['headline'], keep='last')
        df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
        df = df.dropna(subset=['scraped_at'])
        df = df.sort_values(by='scraped_at', ascending=False).reset_index(drop=True)
    
    return df

@st.cache_resource
def load_model():
    """Loads the trained XGBoost model."""
    try:
        model = joblib.load('xgboost_volatility_model.joblib')
        return model
    except FileNotFoundError:
        return None

# --- Data Processing Pipeline (cached) ---

@st.cache_data(ttl="1h")
def run_full_pipeline():
    """
    This function runs the entire data processing pipeline from loading news to the final daily_df.
    """
    df = load_and_combine_data()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --- Download and Prepare Stock Data ---
    tickers = df['ticker'].unique().tolist()
    start_date = '2022-01-01'
    
    stock_df_raw = yf.download(tickers, start=start_date, auto_adjust=True)
    stock_df = stock_df_raw['Close'].stack().reset_index()
    stock_df.columns = ['Date', 'ticker', 'Close']
    
    stock_df = stock_df.sort_values(by=['ticker', 'Date'])
    stock_df['daily_return'] = stock_df.groupby('ticker')['Close'].pct_change()
    stock_df['volatility'] = stock_df.groupby('ticker')['daily_return'].rolling(window=7).std().reset_index(0,drop=True)

    # --- Perform Sentiment Analysis ---
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    def get_sentiment_score(sentiment_result):
        if sentiment_result['label'] == 'positive': return sentiment_result['score']
        elif sentiment_result['label'] == 'negative': return -sentiment_result['score']
        return 0.0

    df['headline'] = df['headline'].fillna('')
    df['summary'] = df['summary'].fillna('')
    df['text_for_sentiment'] = df['headline'] + '. ' + df['summary']
    
    all_sentiments = sentiment_pipeline(df['text_for_sentiment'].tolist())
    df['sentiment_score'] = [get_sentiment_score(s) for s in all_sentiments]

    # --- Merge and Create Final Features ---
    df['date'] = pd.to_datetime(df['scraped_at']).dt.date
    stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
    master_df = pd.merge(df, stock_df, on=['date', 'ticker'], how='inner')

    daily_df = master_df.groupby(['date', 'ticker']).agg(
        mean_sentiment=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        news_volume=('headline', 'count'),
        Close=('Close', 'first'),
        volatility=('volatility', 'first')
    ).reset_index()
    daily_df['sentiment_std'] = daily_df['sentiment_std'].fillna(0)

    for i in range(1, 4):
        daily_df[f'mean_sentiment_lag_{i}'] = daily_df.groupby('ticker')['mean_sentiment'].shift(i)
        daily_df[f'sentiment_std_lag_{i}'] = daily_df.groupby('ticker')['sentiment_std'].shift(i)
        daily_df[f'news_volume_lag_{i}'] = daily_df.groupby('ticker')['news_volume'].shift(i)
    
    daily_df = daily_df.dropna()
    
    return daily_df, df # Return both daily and all_news data

# --- Main App Execution ---

st.title("Financial Sentiment & Volatility Prediction")
model = load_model()
daily_df, all_news_df = run_full_pipeline()

if daily_df.empty or model is None:
    st.error("Data processing failed or model not found. Please check the data sources and ensure the model file is in the repository.")
else:
    # --- Sidebar and UI ---
    st.sidebar.title("Controls")
    tickers = daily_df['ticker'].unique().tolist()
    selected_ticker = st.sidebar.selectbox('Select a Stock Ticker', tickers)

    ticker_daily_df = daily_df[daily_df['ticker'] == selected_ticker].copy()
    ticker_news_df = all_news_df[all_news_df['ticker'] == selected_ticker].copy()

    st.header(f"Dashboard for {selected_ticker}")

    # --- Display Latest Prediction ---
    st.subheader("Live Volatility Forecast")
    latest_data = ticker_daily_df.sort_values('date').tail(1).copy()

    if not latest_data.empty:
        feature_cols = [col for col in latest_data.columns if col not in ['date', 'ticker', 'Close', 'volatility']]
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
    
    # --- Visualizations and News Display ---
    # (This part of the UI remains the same as before)
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

    st.header("Recent News Articles")
    recent_news = ticker_news_df.sort_values('scraped_at', ascending=False).head(5)
    for index, row in recent_news.iterrows():
        st.markdown(f"**{row['headline']}**")
        st.caption(f"Scraped on: {pd.to_datetime(row['scraped_at']).strftime('%Y-%m-%d %H:%M')}")
        if 'sentiment_score' in row and pd.notna(row['sentiment_score']):
            st.markdown(f"Sentiment Score: **{row['sentiment_score']:.2f}**")
        if 'summary' in row and pd.notna(row['summary']):
            st.markdown(f"> {row['summary']}")
        st.markdown("---")