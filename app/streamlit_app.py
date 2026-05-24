import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import generate_features
from src.models import train_models, evaluate_models
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Quantlens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Quantlens")
st.markdown("**Quantitative Finance ML Framework** • Predict 5-day stock returns using advanced ML models")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Stock selection
stock_symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL",
    help="e.g., AAPL, GOOG, MSFT, TSLA"
)

# Date range
cols = st.sidebar.columns(2)
with cols[0]:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2023, 1, 1)
    )
with cols[1]:
    end_date = st.date_input(
        "End Date",
        value=datetime.now()
    )

# Model selection
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    options=["Linear Regression", "Random Forest", "XGBoost"],
    default=["XGBoost"]
)

# Fetch data button
if st.sidebar.button("📥 Fetch & Analyze Data", use_container_width=True):
    with st.spinner("Fetching data and training models..."):
        try:
            # Download stock data
            st.info(f"Downloading data for {stock_symbol}...")
            df = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)

            # Normalize MultiIndex columns from yfinance for single-symbol downloads
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                try:
                    second_level = df.columns.get_level_values(1)
                    if second_level.nunique() == 1:
                        df.columns = df.columns.get_level_values(0)
                    else:
                        st.error("❌ Downloaded data contains multiple tickers; please provide a single symbol.")
                        st.stop()
                except Exception:
                    pass

            if df.empty:
                st.error(f"❌ No data found for {stock_symbol}. Please check the symbol.")
            else:
                # Generate features
                st.info("Generating features...")
                features_df = generate_features(df)
                
                # Prepare data
                X = features_df.drop(['Close', 'target_5d'], axis=1)
                y = features_df['target_5d']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train models
                st.info("Training models...")
                models = train_models(X_train, y_train)
                
                # Store in session state for persistence
                st.session_state['models'] = models
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['features_df'] = features_df
                st.session_state['df'] = df
                st.session_state['stock_symbol'] = stock_symbol
                
                st.success(f"✅ Data loaded and models trained for {stock_symbol}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display content if models are trained
if 'models' in st.session_state:
    models = st.session_state['models']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    features_df = st.session_state['features_df']
    df = st.session_state['df']
    symbol = st.session_state['stock_symbol']
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Indicators", "🤖 Model Performance", "🔮 Predictions", "📊 Feature Analysis"])
    
    # Tab 1: Price and Technical Indicators
    with tab1:
        st.subheader(f"{symbol} Stock Data & Technical Indicators")
        
        # Use a 1D close series for scalar metrics and plotting
        close_series = df['Close'].squeeze()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${close_series.iloc[-1]:.2f}")
        with col2:
            returns = (close_series.iloc[-1] / close_series.iloc[0] - 1) * 100
            st.metric("Period Return", f"{returns:.2f}%")
        with col3:
            volatility = close_series.pct_change().std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{volatility:.2f}%")
        with col4:
            st.metric("Data Points", len(df))
        
        # Price chart with indicators
        fig = go.Figure()
        
        # Close price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=close_series,
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        # SMA indicators
        if 'sma_10' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['sma_10'],
                name='SMA 10',
                line=dict(color='orange', width=1)
            ))
        
        if 'sma_50' in features_df.columns:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df['sma_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ))
        
        fig.update_layout(
            title=f"{symbol} Close Price with Moving Averages",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            if 'rsi_14' in features_df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=features_df.index,
                    y=features_df['rsi_14'],
                    name='RSI 14',
                    line=dict(color='purple')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(
                    title="RSI (Relative Strength Index)",
                    yaxis_title="RSI",
                    xaxis_title="Date",
                    height=300
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            if 'volatility_20' in features_df.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=features_df.index,
                    y=features_df['volatility_20'],
                    name='20-day Volatility',
                    line=dict(color='green')
                ))
                fig_vol.update_layout(
                    title="Rolling Volatility (20-day)",
                    yaxis_title="Volatility",
                    xaxis_title="Date",
                    height=300
                )
                st.plotly_chart(fig_vol, use_container_width=True)
    
    # Tab 2: Model Performance
    with tab2:
        st.subheader("Model Evaluation on Test Set")
        
        # Evaluate models
        from sklearn.metrics import r2_score, mean_squared_error
        
        results = []
        for name, model in models.items():
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            results.append({
                'Model': name.upper(),
                'R² Score': f"{r2:.4f}",
                'RMSE': f"{rmse:.5f}"
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Model comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            r2_scores = []
            model_names = []
            for name, model in models.items():
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                r2_scores.append(r2)
                model_names.append(name.upper())
            
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=model_names,
                y=r2_scores,
                marker_color=['#636EFA', '#EF553B', '#00CC96']
            ))
            fig_r2.update_layout(
                title="R² Score Comparison",
                yaxis_title="R² Score",
                height=350
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            rmse_scores = []
            for name, model in models.items():
                preds = model.predict(X_test)
                rmse = mean_squared_error(y_test, preds) ** 0.5
                rmse_scores.append(rmse)
            
            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Bar(
                x=model_names,
                y=rmse_scores,
                marker_color=['#636EFA', '#EF553B', '#00CC96']
            ))
            fig_rmse.update_layout(
                title="RMSE Comparison",
                yaxis_title="RMSE",
                height=350
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Tab 3: Predictions
    with tab3:
        st.subheader("5-Day Return Predictions")
        
        # Get latest features
        latest_features = X_test.iloc[-1:].values
        
        col1, col2, col3 = st.columns(3)
        
        for idx, (name, model) in enumerate(models.items()):
            pred = model.predict(latest_features)[0]
            
            if idx == 0:
                col = col1
            elif idx == 1:
                col = col2
            else:
                col = col3
            
            with col:
                metric_color = "green" if pred > 0 else "red"
                st.metric(
                    name.upper(),
                    f"{pred:.4f}",
                    f"{pred*100:.2f}%",
                    delta_color="normal"
                )
        
        # Prediction distribution
        st.subheader("Prediction Distribution on Test Set")
        
        all_predictions = {}
        for name, model in models.items():
            preds = model.predict(X_test)
            all_predictions[name.upper()] = preds
        
        fig = go.Figure()
        for name, preds in all_predictions.items():
            fig.add_trace(go.Histogram(
                x=preds,
                name=name,
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Distribution of 5-Day Return Predictions",
            xaxis_title="Predicted Return",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Feature Analysis
    with tab4:
        st.subheader("Feature Statistics")
        
        # Display feature statistics
        feature_stats = X_test.describe().T
        st.dataframe(feature_stats, use_container_width=True)
        
        st.subheader("Feature Correlation with Target")
        
        # Calculate correlations
        temp_df = X_test.copy()
        temp_df['target'] = y_test.values
        correlations = temp_df.corr()['target'].drop('target').sort_values(ascending=False)
        
        fig = go.Figure()
        colors = ['green' if x > 0 else 'red' for x in correlations.values]
        fig.add_trace(go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            marker_color=colors
        ))
        fig.update_layout(
            title="Feature Correlation with 5-Day Returns",
            xaxis_title="Correlation Coefficient",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome message when no data loaded
    st.info("👉 **Get Started:** Enter a stock symbol in the sidebar and click 'Fetch & Analyze Data' to begin")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📥 Data Pipeline")
        st.write("Fetch historical OHLCV data from Yahoo Finance and S&P 500 universe")
    with col2:
        st.markdown("### 🎯 Feature Engineering")
        st.write("Generate 10+ technical indicators including RSI, MACD, Stochastic, and more")
    with col3:
        st.markdown("### 🤖 ML Models")
        st.write("Train ensemble of Linear Regression, Random Forest, and XGBoost")
    
    st.markdown("---")
    st.markdown("""
    ### Sample Stocks to Try
    - **AAPL** - Apple Inc.
    - **GOOG** - Alphabet Inc.
    - **MSFT** - Microsoft Corporation
    - **TSLA** - Tesla Inc.
    - **AMZN** - Amazon.com Inc.
    
    ### Features
    - **RSI 14**: Momentum indicator (0-100 scale)
    - **MACD**: Trend-following momentum indicator
    - **Stochastic**: Price momentum from close relative to range
    - **SMA**: Simple Moving Averages (10 & 50 day)
    - **Volatility**: 20-day rolling standard deviation
    - **Momentum**: 10-day price momentum
    """)

st.markdown("---")
st.markdown("""
<footer style='text-align: center; color: gray; font-size: 12px'>
Quantlens | Quantitative Finance ML Framework | Educational & Research Purposes Only
</footer>
""", unsafe_allow_html=True)
