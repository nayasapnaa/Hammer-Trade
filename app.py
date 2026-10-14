import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# --- PAGE SETUP ---
st.set_set_config = st.set_page_config(page_title="Nifty Live Backtester", layout="wide")
st.title("📊 Nifty 50 Live Historical Strategy Tester")
st.write("Select a custom historical time window from the calendar to instantly fetch data and run the backtester.")

# --- SIDEBAR: DYNAMIC DATE & STRATEGY CONTROLS ---
st.sidebar.header("1. Choose Time Duration")

max_date = datetime.today().date()
min_date = max_date - timedelta(days=20) 

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=max_date - timedelta(days=59), max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=max_date - timedelta(days=59), max_value=max_date)

st.sidebar.header("2. Strategy Configurations")
trade_setup = st.sidebar.selectbox("Candlestick Trigger Style", ["All Patterns (High Frequency)", "Only Hammers/Stars", "Only Engulfing"])
risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio (RRR)", 1.0, 4.0, 1.5, 0.5)
ema_length = st.sidebar.number_input("EMA Length", min_value=5, max_value=50, value=20)
vol_multiplier = st.sidebar.slider("Volume Filter (Lower = More Trades)", 0.0, 2.0, 0.4, 0.1)

# --- BACKTEST CORE ENGINE ---
def process_data(df, rrr, ema_len, vol_mult, setup_mode):
    # CRITICAL FIX: Clean and un-nest Yahoo Finance Column levels properly
    clean_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            clean_cols.append(str(col[0]).lower())
        else:
            clean_cols.append(str(col).lower())
    df.columns = clean_cols
    
    needed_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = list(df.columns)
    
    if not all(col in available_cols for col in needed_cols):
        st.error(f"Data mapping error. Extracted columns were: {available_cols}")
        return df, pd.DataFrame()
        
    df = df[needed_cols].dropna().copy()
    
    for col in needed_cols:
        df[col] = df[col].astype(float)
        
    # Technical Indicators Calculations
    df['ema_20'] = df['close'].ewm(span=ema_len, adjust=False).mean()
    df['date_group'] = df.index.date
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    df['cum_tp_vol'] = (typical_price * df['volume']).groupby(df['date_group']).cumsum()
    df['cum_vol'] = df['volume'].groupby(df['date_group']).cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    
    df['vol_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_spike'] = df['volume'] > (df['vol_sma'] * vol_mult)
    
    # Extract Raw NumPy Arrays
    open_arr = df['open'].to_numpy()
    high_arr = df['high'].to_numpy()
    low_arr = df['low'].to_numpy()
    close_arr = df['close'].to_numpy()
    
    body = np.abs(close_arr - open_arr)
    candle_range = high_arr - low_arr
    
    prev_open = np.zeros_like(open_arr)
    prev_close = np.zeros_like(close_arr)
    prev_open[1:] = open_arr[:-1]
    prev_close[1:] = close_arr[:-1]
    
    # Pattern Logic
    bull_eng = (close_arr > open_arr) & (prev_close < prev_open) & (open_arr <= prev_close) & (close_arr >= prev_open)
    bear_eng = (close_arr < open_arr) & (prev_close > prev_open) & (open_arr >= prev_close) & (close_arr <= prev_open)
    
    lower_shadow = np.where(close_arr > open_arr, open_arr - low_arr, close_arr - low_arr)
    upper_shadow = np.where(close_arr > open_arr, high_arr - close_arr, high_arr - open_arr)
    
    bull_hammer = (lower_shadow > (2 * body)) & (upper_shadow < (0.5 * body)) & (candle_range > 0)
    bear_star = (upper_shadow > (2 * body)) & (lower_shadow < (0.5 * body)) & (candle_range > 0)
    
    if setup_mode == "Only Engulfing":
        df['is_bull'], df['is_bear'] = bull_eng, bear_eng
    elif setup_mode == "Only Hammers/Stars":
        df['is_bull'], df['is_bear'] = bull_hammer, bear_star
    else:
        df['is_bull'], df['is_bear'] = (bull_eng | bull_hammer), (bear_eng | bear_star)

    trade_log = []
    in_position = False
    pos_type, entry_price, sl_price, tp_price, entry_time = None, 0, 0, 0, None
    candle_count = 0
    df['signal'] = "HOLD"
    
    for i in range(1, len(df)):
        current_time = df.index[i]
        current_close = df.iloc[i]['close']
        
        if in_position:
            candle_count += 1
            if pos_type == "CE":
                if df.iloc[i]['low'] <= sl_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": sl_price, "P&L Points": round(sl_price - entry_price, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                elif df.iloc[i]['high'] >= tp_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": tp_price, "P&L Points": round(tp_price - entry_price, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                elif candle_count >= 5:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": current_close, "P&L Points": round(current_close - entry_price, 2), "Outcome": "TIME OUT"})
                    in_position = False
            elif pos_type == "PE":
                if df.iloc[i]['high'] >= sl_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": sl_price, "P&L Points": round(entry_price - sl_price, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                elif df.iloc[i]['low'] <= tp_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": tp_price, "P&L Points": round(entry_price - tp_price, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                elif candle_count >= 5:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": current_close, "P&L Points": round(entry_price - current_close, 2), "Outcome": "TIME OUT"})
                    in_position = False
            continue
            
        current_ema = df.iloc[i]['ema_20']
        if df.iloc[i]['is_bull'] and (current_close > current_ema) and df.iloc[i]['volume_spike']:
            df.iloc[i, df.columns.get_loc('signal')] = "BUY CALL (CE)"
            in_position, pos_type, entry_price, sl_price = True, "CE", current_close, df.iloc[i]['low']
            tp_price = entry_price + ((entry_price - sl_price) * rrr)
            entry_time, candle_count = current_time, 0
        elif df.iloc[i]['is_bear'] and (current_close < current_ema) and df.iloc[i]['volume_spike']:
            df.iloc[i, df.columns.get_loc('signal')] = "BUY PUT (PE)"
            in_position, pos_type, entry_price, sl_price = True, "PE", current_close, df.iloc[i]['high']
            tp_price = entry_price - ((sl_price - entry_price) * rrr)
            entry_time, candle_count = current_time, 0
            
    return df, pd.DataFrame(trade_log)

# --- LIVE API DATA FETCH TRIGGER ---
if start_date > end_date:
    st.error("Error: Start Date must be prior to End Date.")
else:
    ticker_symbol = "^NSEI"
    
    with st.spinner(f"Connecting to live exchanges to download Nifty data..."):
        raw_data = yf.download(tickers=ticker_symbol, start=start_date, end=end_date + timedelta(days=1), interval="5m")
        
    if not raw_data.empty:
        results_df, trades_df = process_data(raw_data.copy(), risk_reward_ratio, ema_length, vol_multiplier, trade_setup)
        
        if not trades_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            win_rate = (len(trades_df[trades_df['P&L Points'] > 0]) / len(trades_df)) * 100
            
            col1.metric("Total Executed Trades", len(trades_df))
            col2.metric("Win Rate (%)", f"{win_rate:.1f}%")
            col3.metric("Net Strategy Points", f"{trades_df['P&L Points'].sum():+.2f}")
            col4.metric("Trading Candles Processed", f"{len(results_df)}")
            
            st.subheader("📋 Closed Trade Performance Sheet")
            def style_pnl(val):
                return f"color: {'green' if val > 0 else 'red'}; font-weight: bold;"
            
            st.dataframe(trades_df.style.map(style_pnl, subset=['P&L Points']))
            
            st.subheader("📈 Technical Chart Overlay")
            fig = go.Figure(data=[go.Candlestick(x=results_df.index, open=results_df['open'], high=results_df['high'], low=results_df['low'], close=results_df['close'], name='Nifty 50')])
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['vwap'], line=dict(color='orange', width=1.2), name='VWAP'))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['ema_20'], line=dict(color='blue', width=1.2), name='EMA'))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"⚠️ Data parsed ({len(results_df)} candles), but no individual setups triggered a trade. Slide your Volume Filter in the sidebar to 0.0 to check candlestick matching directly.")
    else:
        st.info("No data fetched. Ensure your date boundaries fall within the trailing 60 days.")
