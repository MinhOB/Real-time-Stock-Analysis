# Import thư viện
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime.lime_tabular
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
from transformers import pipeline
from statsmodels.tsa.arima.model import ARIMA


# Thiết lập giao diện Streamlit
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("📁 Stock Selection")
    
    ticker = st.selectbox("Select a Ticker:", [
        "Apple (AAPL)", "Amazon (AMZN)", "Google (GOOGL)", "NVIDIA (NVDA)",
        "Microsoft (MSFT)", "Meta (META)", "Intel (INTC)", "Qualcomm (QCOM)", "Tesla (TSLA)"
    ])
    ticker_symbol = ticker.split()[1].strip("()")  # Lấy mã ticker

    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", value=datetime.now())

    date_diff = (end_date - start_date).days
    if date_diff < 50:
        st.error(f"The selected date range is {date_diff} days, which is too short. Please select a date range of at least 50 days to compute indicators like SMA50.")
        st.stop()

    data_freq = st.selectbox("Data Frequency", ["1m", "5m", "15m", "1h", "1d", "1w"], index=3)  # Mặc định là "1h"
    
    chart_type = st.selectbox("Chart Type", ["Candle", "Line", "Step", "Mountain", "Wave", "Scatter", "Bar", "Histogram"], index=0)
    
    st.write("TECHNICAL INDICATORS OPTIONS")

    # Danh sách các tùy chọn
    indicator_options = ["SMA20", "SMA50", "EMA20", "EMA50", "RSI", "MACD", "Bollinger Bands"]

    # Hiển thị multiselect với mặc định là SMA20 và SMA50
    tech_indicators = st.multiselect(
        "Choose Technical Indicators:",
        options=indicator_options,
        default=["SMA20", "SMA50"]
    )

# Attention Layer cho LSTM
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e, axis=-1)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context, alpha

# Hàm lấy dữ liệu từ Yahoo Finance
def load_data_yfinance(symbol, interval='1d', start_date=None, end_date=None):
    try:
        today = datetime.now().date()
        if start_date > today or end_date > today:
            st.error("Start or end date cannot be in the future!")
            return None
        
        if start_date >= end_date:
            st.error("Start date must be earlier than end date!")
            return None

        # Kiểm tra giới hạn khoảng thời gian dựa trên tần suất
        date_diff = (end_date - start_date).days
        interval_limits = {
            '1m': 7,
            '5m': 60,
            '15m': 60,
            '1h': 730,
            '1d': None,  
            '1w': None   
        }
        
        max_days = interval_limits.get(interval)
        if max_days is not None and date_diff > max_days:
            st.warning(f"Interval '{interval}' only supports data up to {max_days} days. Adjusting start date to fit this limit.")
            start_date = end_date - timedelta(days=max_days)

        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date + timedelta(days=1), interval=interval)
        
        if data.empty:
            st.error(f"No data available for {symbol} with interval '{interval}' in this date range! Please check the ticker, interval, or date range.")
            return None
        
        if 'Close' not in data.columns:
            st.error(f"Data for {symbol} does not contain 'Close' column. Please check the data source.")
            return None
        
        if data['Close'].isna().all():
            st.error(f"All 'Close' prices for {symbol} are NaN in this date range. Please select a different date range or ticker.")
            return None
        
        data = data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        data['Adj Close'] = data['Close']
        
        data['% Change'] = data['Close'].pct_change() * 100
        
        data['RSI'] = compute_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])
        data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data['Close'])
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        data = data.dropna()
        
        if data.empty:
            st.error(f"No valid data after processing for {symbol} with interval '{interval}'. Please ensure the data contains valid price information.")
            return None
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance for {symbol} with interval '{interval}': {str(e)}")
        return None

# Hàm tính RSI
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Hàm tính MACD
def compute_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Hàm tính Bollinger Bands
def compute_bollinger_bands(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Hàm lấy tin tức từ NewsAPI
def fetch_news(ticker_symbol):
    ticker_to_company = {
        "AAPL": "Apple", "AMZN": "Amazon", "GOOGL": "Google", "NVDA": "NVIDIA",
        "MSFT": "Microsoft", "META": "Meta", "INTC": "Intel", "QCOM": "Qualcomm", "TSLA": "Tesla"
    }
    company_name = ticker_to_company.get(ticker_symbol, ticker_symbol)
    query = f"{ticker_symbol} OR {company_name}"
    NEWS_API_KEY = "a29bdff41651403a9351130aa40fcd20"
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=10"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        if news_data.get("status") != "ok":
            st.error("Error fetching news: " + news_data.get("message", "Unknown error"))
            return []
        articles = news_data.get("articles", [])
        if not articles:
            st.warning(f"No news found for {ticker_symbol} or {company_name}.")
        return articles
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# Hàm phân tích cảm xúc với transformers
def analyze_sentiment(text):
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
        label = result['label']
        score = result['score']
        if label == "POSITIVE":
            sentiment = "Positive"
        elif label == "NEGATIVE":
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return sentiment, score
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return "Unknown", 0.0

# Hàm tính các chỉ số đánh giá
def calculate_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    mask_mape = y_true != 0
    if np.any(mask_mape):
        mape = np.mean(np.abs((y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape])) * 100
    else:
        mape = np.nan
    
    denominator = np.abs(y_pred) + np.abs(y_true)
    mask_smape = denominator != 0
    if np.any(mask_smape):
        smape = 100 * np.mean(2 * np.abs(y_pred[mask_smape] - y_true[mask_smape]) / denominator[mask_smape])
    else:
        smape = np.nan
    
    return mae, mse, rmse, r2, mape, smape

# Hàm ICFTS - Counterfactual Explanations
def generate_icfts_explanation(model, scaler, model_data, selected_features, y_feature, time_range=30, threshold_change=0.05):
    try:
        if selected_model == "ARIMA":
            st.warning("ICFTS is not applicable for ARIMA models.")
            return None, None, None
        
        # Chuẩn hóa dữ liệu
        scaled_data = scaler.transform(model_data[[y_feature] + selected_features])
        last_sequence = scaled_data[-time_range:, 1:]  # Bỏ cột y_feature
        
        # Lấy giá trị min/max của dữ liệu gốc (sau khi chuẩn hóa) để áp dụng ràng buộc
        min_values = scaler.transform(model_data[[y_feature] + selected_features].min().values.reshape(1, -1))[:, 1:]
        max_values = scaler.transform(model_data[[y_feature] + selected_features].max().values.reshape(1, -1))[:, 1:]
        
        # Dự đoán ban đầu
        if selected_model in ["LSTM", "Transformer"]:
            current_input = last_sequence.reshape(1, time_range, len(selected_features))
            if selected_model == "LSTM":
                original_pred, _ = model.predict(current_input, verbose=0)
            else:
                original_pred = model.predict(current_input, verbose=0)[0]
            original_pred = original_pred[0, 0]
        else:  # XGBoost
            current_input = last_sequence.flatten().reshape(1, -1)
            original_pred = model.predict(current_input)[0]
        
        # Xử lý riêng cho XGBoost
        if selected_model == "XGBoost":
            counterfactual_input = last_sequence.copy()
            target_change = original_pred * (1 - threshold_change)  # Mục tiêu giảm 5%
            best_counterfactual = counterfactual_input.copy()
            best_pred = original_pred
            best_loss = float('inf')
            
            for _ in range(5000):
                # Giảm độ lớn của nhiễu để tránh thay đổi quá lớn
                perturbed_input = last_sequence + np.random.normal(0, 0.1, last_sequence.shape)
                # Áp dụng ràng buộc trong không gian chuẩn hóa
                perturbed_input = np.clip(perturbed_input, min_values, max_values)
                pred = model.predict(perturbed_input.flatten().reshape(1, -1))[0]
                loss = abs(pred - target_change)
                min_change = max(original_pred * 0.05, 0.03)
                if loss < best_loss and abs(pred - original_pred) > min_change:
                    best_loss = loss
                    best_counterfactual = perturbed_input
                    best_pred = pred
            
            counterfactual_input = best_counterfactual
            counterfactual_pred = best_pred
        
        else:  # LSTM và Transformer
            counterfactual_input = tf.Variable(last_sequence, dtype=tf.float32)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Giảm learning rate
            target_change = original_pred * (1 - threshold_change)
            
            for _ in range(500):
                with tf.GradientTape() as tape:
                    input_reshaped = tf.reshape(counterfactual_input, (1, time_range, len(selected_features)))
                    pred = model(input_reshaped)
                    if isinstance(pred, (list, tuple)):
                        pred = pred[0]
                    pred = pred[0, 0]
                    loss = 10.0 * tf.abs(pred - target_change)
                
                gradients = tape.gradient(loss, counterfactual_input)
                if gradients is None:
                    raise ValueError("Gradients are None. Check model output and input compatibility.")
                optimizer.apply_gradients([(gradients, counterfactual_input)])
                # Áp dụng ràng buộc trong không gian chuẩn hóa
                counterfactual_input.assign(tf.clip_by_value(counterfactual_input, min_values, max_values))
            
            counterfactual_input = counterfactual_input.numpy()
            counterfactual_pred = model(counterfactual_input.reshape(1, time_range, len(selected_features)))
            if isinstance(counterfactual_pred, (list, tuple)):
                counterfactual_pred = counterfactual_pred[0]
            counterfactual_pred = counterfactual_pred[0, 0]
        
        # Kiểm tra và cảnh báo
        min_change = max(original_pred * 0.05, 0.03)
        if abs(counterfactual_pred - original_pred) < min_change:
            st.warning(f"Counterfactual prediction ({counterfactual_pred:.2f}) is too close to original prediction ({original_pred:.2f}). The optimization may not have converged properly.")
        
        # Chuyển ngược lại về không gian gốc
        dummy_y = np.zeros((time_range, 1))
        combined_counterfactual = np.column_stack([dummy_y, counterfactual_input])
        counterfactual_data = scaler.inverse_transform(combined_counterfactual)[:, 1:]
        
        # Áp dụng ràng buộc trong không gian gốc
        feature_indices = {feature: idx for idx, feature in enumerate(selected_features)}
        
        # Ràng buộc cho các cột không thể âm
        for feature in ['High', 'Low', 'Open', 'SMA20', 'SMA50']:
            if feature in feature_indices:
                counterfactual_data[:, feature_indices[feature]] = np.maximum(
                    counterfactual_data[:, feature_indices[feature]], 0
                )
        
        # Ràng buộc cho RSI: [0, 100]
        if 'RSI' in feature_indices:
            counterfactual_data[:, feature_indices['RSI']] = np.clip(
                counterfactual_data[:, feature_indices['RSI']], 0, 100
            )
        
        # Ràng buộc cho % Change: [-100, 100]
        if '% Change' in feature_indices:
            counterfactual_data[:, feature_indices['% Change']] = np.clip(
                counterfactual_data[:, feature_indices['% Change']], -100, 100
            )
        
        # Ràng buộc cho MACD (dựa trên dữ liệu gốc, ví dụ: [-500, 500])
        if 'MACD' in feature_indices:
            macd_min = model_data['MACD'].min()
            macd_max = model_data['MACD'].max()
            counterfactual_data[:, feature_indices['MACD']] = np.clip(
                counterfactual_data[:, feature_indices['MACD']], macd_min, macd_max
            )
        
        # Chuyển đổi dự đoán về không gian gốc
        dummy_features = np.zeros((1, len(selected_features)))
        original_pred_transformed = scaler.inverse_transform(np.column_stack([[original_pred], dummy_features]))[0, 0]
        counterfactual_pred_transformed = scaler.inverse_transform(np.column_stack([[counterfactual_pred], dummy_features]))[0, 0]
        
        return counterfactual_data, original_pred_transformed, counterfactual_pred_transformed
    
    except Exception as e:
        st.error(f"Error generating ICFTS explanation: {str(e)}")
        return None, None, None

# Hàm DAVOTS - Dynamic Attribution for Visualizing Time Series
def generate_davots_explanation(model, scaler, model_data, selected_features, y_feature, time_range=30):
    try:
        if selected_model == "ARIMA":
            st.warning("DAVOTS is not applicable for ARIMA models.")
            return
        
        scaled_data = scaler.transform(model_data[[y_feature] + selected_features])
        last_sequence = scaled_data[-time_range:, 1:]
        
        if selected_model == "LSTM":
            current_input = last_sequence.reshape(1, time_range, len(selected_features))
            _, attention_weights = model.predict(current_input, verbose=0)
            attention_weights = attention_weights[0, :, 0]
            
            plt.figure(figsize=(12, 6))
            plt.imshow(attention_weights.reshape(1, -1), cmap='viridis', aspect='auto')
            plt.colorbar(label='Attention Weight')
            plt.title("DAVOTS: Attention Weights for LSTM")
            plt.xlabel("Timestep")
            plt.ylabel("Attention")
            plt.xticks(np.arange(time_range), [f"t-{time_range-i-1}" for i in range(time_range)], rotation=45)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()
        
        elif selected_model == "Transformer":
            current_input = last_sequence.reshape(1, time_range, len(selected_features))
            _, attn_weights = model.predict(current_input, verbose=0)
            attention_weights = np.mean(attn_weights, axis=1)[0]  # Lấy trung bình qua các head
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(attention_weights, cmap='viridis', xticklabels=[f"t-{time_range-i-1}" for i in range(time_range)],
                        yticklabels=[f"t-{time_range-i-1}" for i in range(time_range)])
            plt.title("DAVOTS: Attention Weights for Transformer")
            plt.xlabel("Timestep (Key)")
            plt.ylabel("Timestep (Query)")
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()
        
        elif selected_model == "XGBoost":
            current_input = last_sequence.flatten().reshape(1, -1)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(current_input)
            shap_values = shap_values.reshape(time_range, len(selected_features))
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(shap_values, cmap='coolwarm', xticklabels=selected_features,
                        yticklabels=[f"t-{time_range-i-1}" for i in range(time_range)])
            plt.title("DAVOTS: SHAP-based Attribution for XGBoost")
            plt.xlabel("Feature")
            plt.ylabel("Timestep")
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()
    
    except Exception as e:
        st.error(f"Error generating DAVOTS explanation: {str(e)}")

# Hàm LIME - Local Interpretable Model-agnostic Explanations
def generate_lime_explanation(model, scaler, model_data, selected_features, y_feature, time_range=30):
    try:
        if selected_model == "ARIMA":
            st.warning("LIME is not applicable for ARIMA models.")
            return
        
        scaled_data = scaler.transform(model_data[[y_feature] + selected_features])
        X_full = scaled_data[:, 1:]
        
        X_lime_train = []
        for i in range(len(X_full) - time_range + 1):
            X_lime_train.append(X_full[i:i + time_range, :].flatten())
        X_lime_train = np.array(X_lime_train)
        
        if len(X_lime_train) == 0:
            raise ValueError("Not enough data to create training samples for LIME.")
        
        X_explain = X_full[-time_range:, :].flatten()
        
        if selected_model in ["LSTM", "Transformer"]:
            def model_predict(x):
                n_samples = x.shape[0]
                x_3d = x.reshape(n_samples, time_range, len(selected_features))
                if selected_model == "LSTM":
                    pred, _ = model.predict(x_3d, verbose=0)
                else:
                    pred = model.predict(x_3d, verbose=0)[0]
                return pred.flatten()
        else:
            model_predict = model.predict
        
        expanded_feature_names = []
        for t in range(time_range):
            for feat in selected_features:
                expanded_feature_names.append(f"{feat}_t-{time_range-t-1}")
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_lime_train,
            feature_names=expanded_feature_names,
            mode="regression"
        )
        
        exp = explainer.explain_instance(
            data_row=X_explain,
            predict_fn=model_predict,
            num_features=5
        )
        
        lime_data = exp.as_list()
        features = [item[0] for item in lime_data]
        weights = [item[1] for item in lime_data]
        
        # Hiển thị Grouped Bar Chart trước
        st.write("###### LIME Grouped Bar Chart")
        positive_features = [f for f, w in zip(features, weights) if w > 0]
        positive_weights = [w for w in weights if w > 0]
        negative_features = [f for f, w in zip(features, weights) if w < 0]
        negative_weights = [abs(w) for w in weights if w < 0]
        
        fig_grouped = go.Figure()
        if positive_features:
            fig_grouped.add_trace(go.Bar(
                x=positive_weights,
                y=positive_features,
                name='Positive Impact',
                orientation='h',
                marker_color='green'
            ))
        if negative_features:
            fig_grouped.add_trace(go.Bar(
                x=negative_weights,
                y=negative_features,
                name='Negative Impact',
                orientation='h',
                marker_color='red'
            ))
        fig_grouped.update_layout(
            title="LIME: Grouped Bar Chart of Feature Impacts",
            xaxis_title="Impact Magnitude",
            yaxis_title="Feature",
            barmode='group'
        )
        st.plotly_chart(fig_grouped)
        
        # Sau đó hiển thị Pie Chart
        st.write("###### LIME Pie Chart (Positive vs Negative Impact)")
        positive_sum = sum(w for w in weights if w > 0)
        negative_sum = abs(sum(w for w in weights if w < 0))
        pie_data = [positive_sum, negative_sum]
        pie_labels = ['Positive Impact', 'Negative Impact']
        fig_pie = go.Figure(data=[
            go.Pie(labels=pie_labels, values=pie_data, hole=0.3)
        ])
        fig_pie.update_layout(title="LIME: Positive vs Negative Impact")
        st.plotly_chart(fig_pie)
    
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")

# Hàm SHAP - SHAP Explanations
def generate_shap_explanation(model, scaler, model_data, selected_features, y_feature, time_range=30):
    try:
        if selected_model == "ARIMA":
            st.warning("SHAP is not applicable for ARIMA models.")
            return
        
        scaled_data = scaler.transform(model_data[[y_feature] + selected_features])
        X_full = scaled_data[:, 1:]
        
        # Chuẩn bị dữ liệu cho SHAP (giới hạn số mẫu)
        X_test = []
        max_samples = min(50, len(X_full) - time_range + 1) 
        for i in range(max_samples):
            X_test.append(X_full[i:i + time_range, :].flatten())
        X_test = np.array(X_test)
        
        if len(X_test) == 0:
            raise ValueError("Not enough data to create test samples for SHAP.")
        
        # Đơn giản hóa hàm dự đoán
        if selected_model in ["LSTM", "Transformer"]:
            def model_predict(x):
                x_3d = x.reshape(-1, time_range, len(selected_features))
                pred = model.predict(x_3d, verbose=0)
                if isinstance(pred, (list, tuple)):
                    return pred[0].flatten()
                return pred.flatten()
            background_samples = X_test[:min(5, len(X_test))]
            explainer = shap.KernelExplainer(model_predict, background_samples)
        else:  # XGBoost
            explainer = shap.TreeExplainer(model)
        
        shap_samples = X_test[-min(20, len(X_test)):]  
        shap_values = explainer.shap_values(shap_samples)
        
        # Nếu shap_values là danh sách, lấy phần tử đầu tiên
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Reshape SHAP values thành (n_samples, time_range, n_features)
        shap_values_reshaped = shap_values.reshape(-1, time_range, len(selected_features))
        
        # Gộp SHAP values theo feature (tính trung bình qua các timestep)
        shap_values_per_feature = np.mean(shap_values_reshaped, axis=1)  # (n_samples, n_features)
        
        # Gộp dữ liệu X_test theo feature (tính trung bình qua các timestep)
        X_test_reshaped = shap_samples.reshape(-1, time_range, len(selected_features))
        X_test_per_feature = np.mean(X_test_reshaped, axis=1)  # (n_samples, n_features)
        
        # 1. Beeswarm Plot
        st.write("###### SHAP Beeswarm Plot (Per Feature)")
        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_values_per_feature,
            X_test_per_feature,
            feature_names=selected_features,  # Chỉ dùng tên feature chính
            plot_type="dot",
            show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()
        
        # 2. Bar Plot cho mẫu cuối
        expanded_feature_names = []
        for t in range(time_range):
            for feat in selected_features:
                expanded_feature_names.append(f"{feat}_t-{time_range-t-1}")
        
        st.write("###### SHAP Value for Each Feature (Latest Prediction)")
        shap_values_last = shap_values[-1]  # SHAP values chi tiết cho mẫu cuối
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=expanded_feature_names,
            y=shap_values_last,
            name="SHAP Value",
            marker_color='purple'
        ))
        fig.update_layout(
            title="SHAP Values for Each Feature",
            xaxis_title="Feature",
            yaxis_title="SHAP Value",
            xaxis={'tickangle': 45},
            height=600
        )
        st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")

# Hàm train_and_evaluate (điều chỉnh LSTM và Transformer)
def train_and_evaluate(model_name, df, selected_features, y_feature, train_size, epochs):
    df = df.dropna()
    
    if len(df) < 30:
        raise ValueError(f"Not enough data to train! At least 30 data points are required, but only {len(df)} are available.")
    
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    train_dates = train_data.index
    test_dates = test_data.index
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[y_feature] + selected_features])
    
    time_range = 30
    model = None
    
    all_dates = df.index[time_range:]
    all_predictions = []
    y_true = df[y_feature].iloc[time_range:].values
    
    if model_name == "LSTM":
        X_train = []
        y_train = []
        X_all = []
        
        for i in range(len(train_data) - time_range):
            X_train.append(scaled_data[i:i + time_range, 1:])
            y_train.append(scaled_data[i + time_range, 0])
        
        for i in range(len(df) - time_range):
            X_all.append(scaled_data[i:i + time_range, 1:])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_all = np.array(X_all)
        
        if len(X_train) == 0 or X_train.shape[1] != time_range or X_train.shape[2] != len(selected_features):
            raise ValueError(f"Invalid training data shape: {X_train.shape}, expected (samples, {time_range}, {len(selected_features)})")
        
        # Cấu trúc LSTM
        inputs = Input(shape=(time_range, len(selected_features)))
        lstm1 = LSTM(50, return_sequences=True)(inputs)
        dropout1 = Dropout(0.2)(lstm1)
        context, attention_weights = AttentionLayer()(dropout1)
        dense1 = Dense(20, activation='relu')(context)
        outputs = Dense(1)(dense1)
        model = Model(inputs=inputs, outputs=[outputs, attention_weights])
        
        model.compile(optimizer='adam', loss=['mse', None])
        model.fit(X_train, [y_train, np.zeros((len(y_train), time_range))], epochs=epochs, batch_size=32, verbose=0)
        
        all_predictions = []
        all_attention_weights = []
        for i in range(len(X_all)):
            current_input = X_all[i].reshape(1, time_range, len(selected_features))
            pred, attn = model.predict(current_input, verbose=0)
            all_predictions.append(pred[0, 0])
            all_attention_weights.append(attn[0, :, 0])
        
        dummy_features = np.zeros((len(all_predictions), len(selected_features)))
        combined_predictions = np.column_stack([all_predictions, dummy_features])
        inverse_transformed = scaler.inverse_transform(combined_predictions)
        all_predictions = inverse_transformed[:, 0]
        y_true_transformed = scaler.inverse_transform(np.column_stack([scaled_data[time_range:, 0], dummy_features]))[:, 0]
    
    elif model_name == "Transformer":
        X_train = []
        y_train = []
        X_all = []
        
        for i in range(len(train_data) - time_range):
            X_train.append(scaled_data[i:i + time_range, 1:])
            y_train.append(scaled_data[i + time_range, 0])
        
        for i in range(len(df) - time_range):
            X_all.append(scaled_data[i:i + time_range, 1:])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_all = np.array(X_all)
        
        if len(X_train) == 0 or X_train.shape[1] != time_range or X_train.shape[2] != len(selected_features):
            raise ValueError(f"Invalid training data shape: {X_train.shape}, expected (samples, {time_range}, {len(selected_features)})")
        
        # Cấu trúc Transformer
        inputs = Input(shape=(time_range, len(selected_features)))
        attn_output, attn_weights = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs, return_attention_scores=True)
        x = LayerNormalization(epsilon=1e-6)(attn_output + inputs)
        ffn = Dense(64, activation='relu')(x)
        ffn = Dense(len(selected_features))(ffn)
        x = LayerNormalization(epsilon=1e-6)(ffn + x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(20, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=[outputs, attn_weights])
        
        model.compile(optimizer='adam', loss=['mse', None])
        model.fit(X_train, [y_train, np.zeros((len(y_train), 4, time_range, time_range))], epochs=epochs, batch_size=32, verbose=0)
        
        all_predictions = []
        all_attention_weights = []
        for i in range(len(X_all)):
            current_input = X_all[i].reshape(1, time_range, len(selected_features))
            pred, attn = model.predict(current_input, verbose=0)
            all_predictions.append(pred[0, 0])
            all_attention_weights.append(attn[0])
        
        dummy_features = np.zeros((len(all_predictions), len(selected_features)))
        combined_predictions = np.column_stack([all_predictions, dummy_features])
        inverse_transformed = scaler.inverse_transform(combined_predictions)
        all_predictions = inverse_transformed[:, 0]
        y_true_transformed = scaler.inverse_transform(np.column_stack([scaled_data[time_range:, 0], dummy_features]))[:, 0]
    
    elif model_name == "XGBoost":
        X_train = []
        y_train = []
        X_all = []
        
        for i in range(len(train_data) - time_range):
            X_train.append(scaled_data[i:i + time_range, 1:].flatten())
            y_train.append(scaled_data[i + time_range, 0])
        
        for i in range(len(df) - time_range):
            X_all.append(scaled_data[i:i + time_range, 1:].flatten())
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_all = np.array(X_all)
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        all_predictions = model.predict(X_all)
        dummy_features = np.zeros((len(all_predictions), len(selected_features)))
        combined_predictions = np.column_stack([all_predictions, dummy_features])
        inverse_transformed = scaler.inverse_transform(combined_predictions)
        all_predictions = inverse_transformed[:, 0]
        y_true_transformed = scaler.inverse_transform(np.column_stack([scaled_data[time_range:, 0], dummy_features]))[:, 0]
        all_attention_weights = None
    
    elif model_name == "ARIMA":
        # Kiểm tra dữ liệu đầu vào
        if df[y_feature].isna().any():
            raise ValueError("ARIMA data contains NaN values. Please ensure the data is clean.")
        if len(df) < time_range:
            raise ValueError(f"ARIMA data is too short: {len(df)} samples. At least {time_range} samples are required.")
        
        # Fit mô hình trên toàn bộ dữ liệu từ đầu đến len(df) - time_range
        train_series = df[y_feature].iloc[:len(df) - time_range]
        model = ARIMA(train_series, order=(5, 1, 0))
        try:
            fitted_model = model.fit()
        except Exception as e:
            raise ValueError(f"Failed to fit ARIMA model: {str(e)}")
        
        # Dự đoán từ time_range đến len(df) - 1
        all_predictions = fitted_model.predict(start=time_range, end=len(df) - 1, typ='levels').values
        
        y_true_transformed = y_true
        scaler = None
        all_attention_weights = None
        model = fitted_model
    
    return all_dates, all_predictions, y_true_transformed, model, scaler, all_attention_weights

def predict_future(model_name, df, selected_features, y_feature, model, scaler, forecast_horizon, confidence_level=0.95):
    df = df.dropna()
    time_range = 30
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')
    
    if model_name == "ARIMA":
        model_fit = model
        forecast = model_fit.forecast(steps=forecast_horizon)
        predictions = forecast.values
        forecast_obj = model_fit.get_forecast(steps=forecast_horizon)
        conf_int = forecast_obj.conf_int(alpha=1 - confidence_level)
        lower_bound = conf_int.iloc[:, 0].values
        upper_bound = conf_int.iloc[:, 1].values
    else:
        scaled_data = scaler.transform(df[[y_feature] + selected_features])
        
        if model_name in ["LSTM", "Transformer"]:
            last_sequence = scaled_data[-time_range:, 1:]
            predictions = []
            temp_sequence = last_sequence.copy()
            
            all_predictions = []
            for _ in range(50):
                temp_sequence = last_sequence.copy()
                preds = []
                for _ in range(forecast_horizon):
                    current_input = temp_sequence.reshape(1, time_range, len(selected_features))
                    if model_name == "LSTM":
                        next_y_pred, _ = model.predict(current_input, verbose=0)
                    else:
                        next_y_pred = model.predict(current_input, verbose=0)[0]
                    next_y_pred = next_y_pred[0, 0]
                    preds.append(next_y_pred)
                    temp_sequence = np.roll(temp_sequence, -1, axis=0)
                    temp_sequence[-1, 0] = next_y_pred + np.random.normal(0, 0.01)
                all_predictions.append(preds)
            
            all_predictions = np.array(all_predictions)
            predictions = np.mean(all_predictions, axis=0)
            std_predictions = np.std(all_predictions, axis=0)
            z_score = 1.96 if confidence_level == 0.95 else 2.58
            lower_bound = predictions - z_score * std_predictions
            upper_bound = predictions + z_score * std_predictions
            
            dummy_features = np.zeros((forecast_horizon, len(selected_features)))
            combined_predictions = np.column_stack([predictions, dummy_features])
            combined_lower = np.column_stack([lower_bound, dummy_features])
            combined_upper = np.column_stack([upper_bound, dummy_features])
            
            predictions = scaler.inverse_transform(combined_predictions)[:, 0]
            lower_bound = scaler.inverse_transform(combined_lower)[:, 0]
            upper_bound = scaler.inverse_transform(combined_upper)[:, 0]
        
        else:
            last_sequence = scaled_data[-time_range:, 1:].flatten()
            predictions = []
            temp_sequence = last_sequence.copy()
            
            all_predictions = []
            for _ in range(50):
                temp_sequence = last_sequence.copy()
                preds = []
                for _ in range(forecast_horizon):
                    current_input = temp_sequence.reshape(1, -1)
                    next_y_pred = model.predict(current_input)[0]
                    preds.append(next_y_pred)
                    temp_sequence = np.roll(temp_sequence, -len(selected_features))
                    temp_sequence[-len(selected_features):] = next_y_pred + np.random.normal(0, 0.01)
                all_predictions.append(preds)
            
            all_predictions = np.array(all_predictions)
            predictions = np.mean(all_predictions, axis=0)
            std_predictions = np.std(all_predictions, axis=0)
            z_score = 1.96 if confidence_level == 0.95 else 2.58
            lower_bound = predictions - z_score * std_predictions
            upper_bound = predictions + z_score * std_predictions
            
            dummy_features = np.zeros((forecast_horizon, len(selected_features)))
            combined_predictions = np.column_stack([predictions, dummy_features])
            combined_lower = np.column_stack([lower_bound, dummy_features])
            combined_upper = np.column_stack([upper_bound, dummy_features])
            
            predictions = scaler.inverse_transform(combined_predictions)[:, 0]
            lower_bound = scaler.inverse_transform(combined_lower)[:, 0]
            upper_bound = scaler.inverse_transform(combined_upper)[:, 0]
    
    return future_dates, predictions, lower_bound, upper_bound

# Lấy dữ liệu cho ticker đã chọn
df = load_data_yfinance(ticker_symbol, interval=data_freq, start_date=start_date, end_date=end_date)

# Main Panel
if df is not None and not df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Pricing Data", "📰 News", "💬 Sentiment Analysis", "💡 Stock Price Prediction"])

    with tab1:
        st.success(f"Data for {ticker_symbol} loaded successfully!")
        
        if df['Close'].isna().all():
            st.error("No valid price data to compute metrics. Please check the data.")
            annual_return = std_dev = risk_adj_return = "N/A"
        else:
            returns = df['Close'].pct_change().dropna()
            if returns.empty:
                st.warning("Not enough data to compute metrics (requires at least 2 days of data).")
                annual_return = std_dev = risk_adj_return = "N/A"
            else:
                annual_factor = 252
                annual_return = returns.mean() * annual_factor * 100
                std_dev = returns.std() * np.sqrt(annual_factor) * 100
                risk_adj_return = annual_return / std_dev if std_dev != 0 else 0
                annual_return = f"{annual_return:.2f}%"
                std_dev = f"{std_dev:.2f}%"
                risk_adj_return = f"{risk_adj_return:.2f}"

        st.write(f"Showing data with '{data_freq}' frequency:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Annual Return", annual_return)
        col2.metric("Standard Deviation", std_dev)
        col3.metric("Risk Adjusted Return", risk_adj_return)

        st.write("### Pricing Data")
        display_df = df[['Close', 'High', 'Low', 'Open', 'Volume', '% Change']].reset_index()
        display_df = display_df.rename(columns={'index': 'Date'})
        display_df['% Change'] = display_df['% Change'].apply(lambda x: f"{x:.3f}%" if not pd.isna(x) else "N/A")
        st.dataframe(display_df)

        st.write("### Data Visualization")
        
        st.write("#### Price Chart")
        if df['Close'].isna().all():
            st.error("No valid price data to plot the chart. Please check the data.")
        else:
            fig_price = go.Figure()
            if chart_type == "Candle":
                fig_price.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Candlestick'
                ))
            elif chart_type == "Line":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    connectgaps=False
                ))
            elif chart_type == "Step":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(shape='hv'),
                    connectgaps=False
                ))
            elif chart_type == "Mountain":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    fill='tozeroy',
                    line=dict(color='blue'),
                    connectgaps=False
                ))
            elif chart_type == "Wave":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines+markers',
                    name='Close Price',
                    line=dict(shape='spline'),
                    connectgaps=False
                ))
            elif chart_type == "Scatter":
                fig_price.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='markers',
                    name='Close Price',
                    connectgaps=False
                ))
            elif chart_type == "Bar":
                fig_price.add_trace(go.Bar(
                    x=df.index,
                    y=df['Close'],
                    name='Close Price',
                    marker_color='blue'
                ))
            elif chart_type == "Histogram":
                fig_price.add_trace(go.Histogram(
                    x=df['Close'],
                    name='Price Distribution',
                    marker_color='blue',
                    opacity=0.75
                ))
                fig_price.update_layout(
                    title=f"Price Distribution for {ticker_symbol}",
                    xaxis_title="Price",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_price)
                st.write("Note: Histogram shows the distribution of closing prices, not a time series.")
                st.stop()

            for indicator in tech_indicators:
                if indicator == "SMA20":
                    fig_price.add_trace(go.Scatter(
                        x=df.index,
                        y=df['SMA20'],
                        mode='lines',
                        name='SMA20',
                        line=dict(color='orange'),
                        connectgaps=False
                    ))
                elif indicator == "SMA50":
                    if 'SMA50' in df.columns and not df['SMA50'].isna().all():
                        fig_price.add_trace(go.Scatter(
                            x=df.index,
                            y=df['SMA50'],
                            mode='lines',
                            name='SMA50',
                            line=dict(color='purple'),
                            connectgaps=False
                        ))
                    else:
                        st.warning("SMA50 data is not available for the selected date range.")
                elif indicator == "EMA20":
                    fig_price.add_trace(go.Scatter(
                        x=df.index,
                        y=df['EMA20'],
                        mode='lines',
                        name='EMA20',
                        line=dict(color='green'),
                        connectgaps=False
                    ))
                elif indicator == "EMA50":
                    fig_price.add_trace(go.Scatter(
                        x=df.index,
                        y=df['EMA50'],
                        mode='lines',
                        name='EMA50',
                        line=dict(color='red'),
                        connectgaps=False
                    ))
                elif indicator == "Bollinger Bands":
                    fig_price.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Bollinger_Upper'],
                        mode='lines',
                        name='Bollinger Upper',
                        line=dict(color='red'),
                        connectgaps=False
                    ))
                    fig_price.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Bollinger_Lower'],
                        mode='lines',
                        name='Bollinger Lower',
                        line=dict(color='green'),
                        connectgaps=False
                    ))

            if "RSI" in tech_indicators:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    connectgaps=False
                ))
                fig_rsi.update_layout(title=f"RSI for {ticker_symbol}", xaxis_title="Date", yaxis_title="RSI")
                st.plotly_chart(fig_rsi)

            if "MACD" in tech_indicators:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    mode='lines',
                    name='MACD',
                    connectgaps=False
                ))
                fig_macd.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='orange'),
                    connectgaps=False
                ))
                fig_macd.update_layout(title=f"MACD for {ticker_symbol}", xaxis_title="Date", yaxis_title="MACD")
                st.plotly_chart(fig_macd)

            if chart_type != "Histogram":
                fig_price.update_layout(title=f"{ticker_symbol} Price ({data_freq})", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_price)

        st.write("#### Volume Chart")
        if df['Volume'].isna().all():
            st.error("No valid volume data to plot the chart. Please check the data.")
        else:
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            fig_volume.update_layout(title=f"Volume for {ticker_symbol}", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig_volume)

    with tab2:
        st.write("### News")
        articles = fetch_news(ticker_symbol)
        if articles:
            for article in articles:
                title = article.get("title", "No title")
                description = article.get("description", "No description")
                url = article.get("url", "#")
                published_at = article.get("publishedAt", "Unknown")
                st.write(f"**{title}**")
                st.write(f"Published at: {published_at}")
                st.write(description)
                st.write(f"[Read more]({url})")
                st.markdown("---")
        else:
            st.write("No news found for this ticker. Please check the API key or try another ticker.")

    with tab3:
        st.write("### Sentiment Analysis")
        articles = fetch_news(ticker_symbol)
        if articles:
            sentiment_data = []
            for article in articles:
                title = article.get("title", "No title")
                description = article.get("description", "No description") or ""
                content = title + " " + description
                sentiment, score = analyze_sentiment(content)
                sentiment_data.append({
                    "Title": title,
                    "Sentiment": sentiment,
                    "Confidence Score": score
                })
            
            sentiment_df = pd.DataFrame(sentiment_data)
            st.write("#### Sentiment Analysis Results")
            st.dataframe(sentiment_df)

            sentiment_counts = sentiment_df["Sentiment"].value_counts()
            st.write("#### Sentiment Distribution")
            fig = go.Figure(data=[
                go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values)
            ])
            st.plotly_chart(fig)
        else:
            st.write("No news found for sentiment analysis. Please check the API key or try another ticker.")

    with tab4:
        st.write("### Stock Price Prediction")

        st.write("#### Select Data Source")
        data_source = st.radio("Data Source", ["Use Pricing Data", "Upload Custom Data"])
        
        if data_source == "Use Pricing Data":
            data = df.copy()
        else:
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                data['% Change'] = data['Close'].pct_change() * 100
                data['RSI'] = compute_rsi(data['Close'])
                data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])
                data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data['Close'])
                data['SMA20'] = data['Close'].rolling(window=20).mean()
                if len(data) >= 50:
                    data['SMA50'] = data['Close'].rolling(window=50).mean()
                else:
                    data['SMA50'] = np.nan
                data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
                data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
                data = data.dropna()
                if data.empty:
                    st.error("Uploaded data is empty after processing. Please upload a valid CSV file with sufficient data.")
                    st.stop()
            else:
                st.warning("Please upload a CSV file to proceed.")
                st.stop()

        st.write("#### Data")
        display_df = data[['Close', 'High', 'Low', 'Open', 'Volume', '% Change', 'SMA20', 'SMA50']].reset_index()
        display_df = display_df.rename(columns={'index': 'Date'})
        display_df['% Change'] = display_df['% Change'].apply(lambda x: f"{x:.3f}%" if not pd.isna(x) else "N/A")
        st.dataframe(display_df)

        st.write("#### Select Input Features")
        all_features = ["High", "Low", "Open", "Volume", "RSI", "MACD", "MACD_Signal", "Bollinger_Upper", "Bollinger_Lower", "SMA20", "SMA50", "EMA20", "EMA50", "% Change"]
        selected_features = st.multiselect("Features", all_features, default=["High", "Low"])
        y_feature = "Close"

        if not selected_features:
            st.warning("Please select at least one feature to proceed!")
            st.stop()

        st.write("#### Select Train-Test Ratio (%)")
        train_ratio = st.slider("Train Ratio", 50, 95, 80, 1)
        train_size = int(len(data) * train_ratio / 100)
        test_size = len(data) - train_size
        st.write(f"Train: {train_size} samples, Test: {test_size} samples")

        st.write("#### Select Model")
        selected_model = st.radio("Model", ["LSTM", "Transformer", "XGBoost", "ARIMA"])

        if selected_model in ["LSTM", "Transformer"]:
            st.write("#### Number of Epochs")
            epochs = st.slider("Epochs", 1, 50, 5, 1)
        else:
            epochs = None

        model_data = data[[y_feature] + selected_features].copy()

        if st.button("Train and Evaluate"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Training {selected_model} model...")
            try:
                all_dates, all_predictions, y_true, model, scaler, all_attention_weights = train_and_evaluate(
                    selected_model, model_data, selected_features, y_feature, train_size, epochs
                )
                st.session_state['all_dates'] = all_dates
                st.session_state['all_predictions'] = all_predictions
                st.session_state['y_true'] = y_true
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['all_attention_weights'] = all_attention_weights
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                mae, mse, rmse, r2, mape, smape = calculate_metrics(y_true, all_predictions)
                st.session_state['metrics'] = {
                    'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mape': mape, 'smape': smape
                }
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=model_data.index,
                    y=model_data[y_feature],
                    mode='lines',
                    name="Actual Price",
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=all_dates,
                    y=all_predictions,
                    mode='lines',
                    name=f"{selected_model} Prediction",
                    line=dict(color='orange', dash='dash')
                ))
                fig.update_layout(title=f"Prediction for {selected_model}", xaxis_title="Date", yaxis_title="Price")
                st.session_state['prediction_chart'] = fig
            
            except Exception as e:
                st.error(f"Error training {selected_model} model: {str(e)}")
                status_text.text("Training failed!")

        if 'metrics' in st.session_state:
            st.write("#### Model Evaluation")
            metrics = st.session_state['metrics']
            st.write(f"MAE: {metrics['mae']:.2f}")
            st.write(f"MSE: {metrics['mse']:.2f}")
            st.write(f"RMSE: {metrics['rmse']:.2f}")
            st.write(f"R²: {metrics['r2']:.2f}")
            st.write(f"MAPE: {metrics['mape']:.2f}%")
            st.write(f"SMAPE: {metrics['smape']:.2f}%")

        if 'prediction_chart' in st.session_state:
            st.write("#### Prediction Chart")
            st.write(f"Prediction for {selected_model}")
            st.plotly_chart(st.session_state['prediction_chart'])

        st.write("#### Multi-step Future Prediction")
        forecast_horizon = st.slider("Number of sessions to predict", 1, 30, 5, 1)
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95, 1) / 100
        
        if st.button("Predict Future Sessions"):
            if 'model' not in st.session_state:
                st.error("Please train the model first!")
            else:
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                
                future_dates, future_predictions, lower_bound, upper_bound = predict_future(
                    selected_model, model_data, selected_features, y_feature, model, scaler, forecast_horizon, confidence_level
                )
                
                st.session_state['future_dates'] = future_dates
                st.session_state['future_predictions'] = future_predictions
                st.session_state['lower_bound'] = lower_bound
                st.session_state['upper_bound'] = upper_bound
                
                st.session_state['next_session_price'] = future_predictions[0]
                
                fig_interval = go.Figure()
                fig_interval.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines',
                    name="Prediction",
                    line=dict(color='blue')
                ))
                fig_interval.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    mode='lines',
                    name="Upper Bound",
                    line=dict(color='red', dash='dash')
                ))
                fig_interval.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    mode='lines',
                    name="Lower Bound",
                    line=dict(color='green', dash='dash')
                ))
                fig_interval.update_layout(title="Prediction Intervals", xaxis_title="Date", yaxis_title="Price")
                st.session_state['interval_chart'] = fig_interval
                
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=model_data.index[-60:],
                    y=model_data[y_feature].iloc[-60:],
                    mode='lines',
                    name="Actual Price",
                    line=dict(color='blue')
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines',
                    name=f"{selected_model} Prediction",
                    line=dict(color='orange', dash='dash')
                ))
                fig_future.update_layout(title=f"Future Prediction for {selected_model}", xaxis_title="Date", yaxis_title="Price")
                st.session_state['future_chart'] = fig_future

        if 'next_session_price' in st.session_state:
            st.write(f"#### Predicted Price for Next Session: {st.session_state['next_session_price']:.2f}")

        if 'lower_bound' in st.session_state and 'upper_bound' in st.session_state:
            st.write("#### Prediction Intervals")
            st.write(f"Confidence Level: {int(confidence_level * 100)}%")
            st.write(f"Interval (Confidence Level {int(confidence_level * 100)}%): [{st.session_state['lower_bound'][0]:.2f}, {st.session_state['upper_bound'][0]:.2f}]")

        if 'interval_chart' in st.session_state:
            st.plotly_chart(st.session_state['interval_chart'])

        if 'future_chart' in st.session_state:
            st.write("#### Future Prediction Chart")
            st.plotly_chart(st.session_state['future_chart'])

        st.write("#### Explain Prediction")
        explain_methods = st.multiselect(
            "Choose methods (you can select multiple)", 
            ["LIME", "SHAP", "ICFTS", "DAVOTS"],
            default=["LIME"]
        )
        
        if st.button("Explain"):
            if 'model' not in st.session_state:
                st.error("Please train the model first!")
            elif not explain_methods:
                st.warning("Please select at least one explanation method!")
            else:
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                all_attention_weights = st.session_state.get('all_attention_weights', None)
                
                for explain_method in explain_methods:
                    if explain_method == "LIME":
                        st.write(f"#### LIME - Local Interpretable Model-agnostic Explanations ({selected_model})")
                        st.write("""
                        **What is LIME?**  
                        LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the model locally with a simpler, interpretable model (e.g., linear regression). It perturbs the input data around a specific instance and observes how the predictions change, identifying which features most influence the prediction in that local region.
                        """)
                        generate_lime_explanation(model, scaler, model_data, selected_features, y_feature)
                    
                    elif explain_method == "SHAP":
                        st.write(f"#### SHAP - SHAP Explanations ({selected_model})")
                        st.write("""
                        **What is SHAP?**  
                        SHAP (SHapley Additive exPlanations) explains individual predictions by assigning each feature an importance value. It is based on cooperative game theory, ensuring a fair distribution of the 'payout' (prediction) among features. SHAP values show how much each feature contributes to pushing the prediction away from a baseline.
                        """)
                        generate_shap_explanation(model, scaler, model_data, selected_features, y_feature)
                    
                    elif explain_method == "ICFTS":
                        st.write(f"#### ICFTS - Counterfactual Explanations ({selected_model})")
                        st.write("""
                        **What is ICFTS?**  
                        ICFTS (Interpretable Counterfactual Explanations for Time Series) provides counterfactual explanations by identifying minimal changes to the input features that would lead to a different prediction. It answers questions like, "What would need to change in the data to alter the model's prediction?" This method is particularly useful for understanding decision boundaries in time series models.
                        """)
                        counterfactual_data, original_pred, counterfactual_pred = generate_icfts_explanation(
                            model, scaler, model_data, selected_features, y_feature
                        )
                        if counterfactual_data is not None:
                            st.write(f"Original Prediction: {original_pred:.2f}")
                            st.write(f"Counterfactual Prediction: {counterfactual_pred:.2f}")
                            st.write("##### Counterfactual Data (last 5 timesteps):")
                            counterfactual_df = pd.DataFrame(
                                counterfactual_data[-5:], 
                                columns=selected_features,
                                index=range(-5, 0)
                            )
                            st.dataframe(counterfactual_df)
                    
                    elif explain_method == "DAVOTS":
                        st.write(f"#### DAVOTS - Dynamic Attribution for Visualizing Time Series ({selected_model})")
                        st.write("""
                        **What is DAVOTS?**  
                        DAVOTS (Dynamic Attribution for Visualizing Time Series) focuses on visualizing the importance of features over time in time series models. For models like LSTM and Transformer, it uses attention mechanisms to highlight which timesteps or features the model focuses on. For XGBoost, it leverages SHAP values to show feature importance across timesteps, providing a dynamic view of the model's decision-making process.
                        """)
                        generate_davots_explanation(model, scaler, model_data, selected_features, y_feature)

else:
    st.error("Unable to load data. Please check the ticker, date range, or network connection!")
