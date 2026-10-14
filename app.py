import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty 50 Strategy Tester", layout="wide")
st.title("📊 Nifty 50 5-Min Options Strategy Tester")
st.write("Upload your 5-minute historical index data, adjust settings, and view triggered signals instantly.")

# --- SIDEBAR INTERFACE CONTROL ---
st.sidebar.header("Strategy Configurations")
risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio (RRR)", 1.0, 4.0, 2.0, 0.5)
vol_multiplier = st.sidebar.slider("Volume Spike Multiplier", 1.0, 3.0, 1.5, 0.1)
ema_length = st.sidebar.number_input("EMA Length", min_value=5, max_value=50, value=20)

# --- BACKEND STRATEGY PROCESSING LOGIC ---
def process_data(df, rrr, vol_mult, ema_len):
    df.columns = [col.lower() for col in df.columns]
    
    # Mathematical Indicators
    df['ema_20'] = ta.ema(df['close'], length=ema_len)
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    df['vol_sma'] = ta.sma(df['volume'], length=10)
    df['volume_spike'] = df['volume'] > (df['vol_sma'] * vol_mult)
    
    # Candle Setup Identification
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    df['bullish_engulfing'] = (df['close'] > df['open']) & (df['prev_close'] < df['prev_open']) & (df['open'] <= df['prev_close']) & (df['close'] >= df['prev_open'])
    df['bearish_engulfing'] = (df['close'] < df['open']) & (df['prev_close'] > df['prev_open']) & (df['open'] >= df['prev_close']) & (df['close'] <= df['prev_open'])
    
    # Initialize Tracking Fields
    df['signal'] = "HOLD"
    df['sl_price'] = np.nan
    df['tp_price'] = np.nan
    
    for i in range(1, len(df)):
        current_close = df.iloc[i]['close']
        current_ema = df.iloc[i]['ema_20']
        current_vwap = df.iloc[i]['vwap']
        
        # Bullish Entry Logic
        if (current_close > current_ema) and (current_close > current_vwap) and df.iloc[i]['bullish_engulfing'] and df.iloc[i]['volume_spike']:
            df.iloc[i, df.columns.get_loc('signal')] = "BUY CALL (CE)"
            df.iloc[i, df.columns.get_loc('sl_price')] = df.iloc[i]['low']
            df.iloc[i, df.columns.get_loc('tp_price')] = current_close + ((current_close - df.iloc[i]['low']) * rrr)
            
        # Bearish Entry Logic
        elif (current_close < current_ema) and (current_close < current_vwap) and df.iloc[i]['bearish_engulfing'] and df.iloc[i]['volume_spike']:
            df.iloc[i, df.columns.get_loc('signal')] = "BUY PUT (PE)"
            df.iloc[i, df.columns.get_loc('sl_price')] = df.iloc[i]['high']
            df.iloc[i, df.columns.get_loc('tp_price')] = current_close - ((df.iloc[i]['high'] - current_close) * rrr)
            
    return df

# --- SIMULATOR / DATA LOADER ---
uploaded_file = st.file_uploader("Upload Nifty 50 5-Min CSV File (Columns must include: Date, Open, High, Low, Close, Volume)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Flexible parsing for common date column names
    date_col = [c for c in data.columns if 'date' in c.lower() or 'time' in c.lower()][0]
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    
    # Run Engine
    with st.spinner("Processing technical indicators and scanning candles..."):
        results = process_data(data.copy(), risk_reward_ratio, vol_multiplier, ema_length)
    
    trades = results[results['signal'] != "HOLD"]
    
    # Layout Split
    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric("Total Trades Found", len(trades))
        st.metric("Call (CE) Signals", len(trades[trades['signal'] == "BUY CALL (CE)"]))
        st.metric("Put (PE) Signals", len(trades[trades['signal'] == "BUY PUT (PE)"]))
    
    with col2:
        # Plotly Candlestick Graphing Generation
        fig = go.Figure(data=[go.Candlestick(x=results.index, open=results['open'], high=results['high'], low=results['low'], close=results['close'], name='Nifty 50')])
        fig.add_trace(go.Scatter(x=results.index, y=results['vwap'], line=dict(color='orange', width=1.5), name='VWAP'))
        fig.add_trace(go.Scatter(x=results.index, y=results['ema_20'], line=dict(color='blue', width=1.5), name='EMA 20'))
        
        # Overlay buy points onto map
        ce_signals = trades[trades['signal'] == "BUY CALL (CE)"]
        pe_signals = trades[trades['signal'] == "BUY PUT (PE)"]
        
        fig.add_trace(go.Scatter(x=ce_signals.index, y=ce_signals['close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='CE Buy Signal'))
        fig.add_trace(go.Scatter(x=pe_signals.index, y=pe_signals['close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='PE Buy Signal'))
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
    # Trade Summary Output
    st.subheader("📋 Triggered Signals Log")
    st.dataframe(trades[['close', 'signal', 'sl_price', 'tp_price']])
else:
    st.info("💡 Awaiting CSV historical tracking upload. Please upload data to activate simulation analysis graphs.")
