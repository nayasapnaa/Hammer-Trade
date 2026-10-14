import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty Live Backtester", layout="wide")
st.title("🦅 Nifty 50 Live Historical Strategy Tester")
st.write("Select a custom historical time window from the calendar to instantly fetch data and run the backtester.")

# --- SIDEBAR: DYNAMIC DATE & STRATEGY CONTROLS ---
st.sidebar.header("1. Choose Time Duration")

# Set default date limits (yfinance provides 5-min intraday data up to the last 60 days)
max_date = datetime.today().date()
min_date = max_date - timedelta(days=59)

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

st.sidebar.header("2. Strategy Configurations")
trade_setup = st.sidebar.selectbox("Candlestick Trigger Style", ["All Patterns (High Frequency)", "Only Hammers/Stars", "Only Engulfing"])
risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio (RRR)", 1.0, 4.0, 1.5, 0.5)
ema_length = st.sidebar.number_input("EMA Length", min_value=5, max_value=50, value=20)
vol_multiplier = st.sidebar.slider("Volume Filter (Lower = More Trades)", 0.2, 2.0, 0.7, 0.1)

# --- BACKTEST CORE ENGINE ---
def process_data(df, rrr, ema_len, vol_mult, setup_mode):
    df.columns = [col.lower() for col in df.columns]
    
    # Technical Indicators
    df['ema_20'] = df['close'].ewm(span=ema_len, adjust=False).mean()
    df['date_group'] = df.index.date
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    df['cum_tp_vol'] = (typical_price * df['volume']).groupby(df['date_group']).cumsum()
    df['cum_vol'] = df['volume'].groupby(df['date_group']).cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    
    df['vol_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_spike'] = df['volume'] > (df['vol_sma'] * vol_mult)
    
    # Candle Dimensions
    body = (df['close'] - df['open']).abs()
    candle_range = df['high'] - df['low']
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    # Pattern Logic
    bull_eng = (df['close'] > df['open']) & (df['prev_close'] < df['prev_open']) & (df['open'] <= df['prev_close']) & (df['close'] >= df['prev_open'])
    bear_eng = (df['close'] < df['open']) & (df['prev_close'] > df['prev_open']) & (df['open'] >= df['prev_close']) & (df['close'] <= df['prev_open'])
    
    lower_shadow = np.where(df['close'] > df['open'], df['open'] - df['low'], df['close'] - df['low'])
    upper_shadow = np.where(df['close'] > df['open'], df['high'] - df['close'], df['high'] - df['open'])
    
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
    # yfinance uses ^NSEI ticker symbol for the Nifty 50 Index
    ticker_symbol = "^NSEI"
    
    with st.spinner(f"Connecting to live exchanges to download Nifty data from {start_date} to {end_date}..."):
        # Fetching 5-minute data chunks from Yahoo Finance
        raw_data = yf.download(tickers=ticker_symbol, start=start_date, end=end_date + timedelta(days=1), interval="5m")
        
    if not raw_data.empty:
        # Convert MultiIndex columns if present in yfinance outputs
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
            
        results_df, trades_df = process_data(raw_data.copy(), risk_reward_ratio, ema_length, vol_multiplier, trade_setup)
        
        if not trades_df.empty:
            # Metrics Row Display
            col1, col2, col3, col4 = st.columns(4)
            win_rate = (len(trades_df[trades_df['P&L Points'] > 0]) / len(trades_df)) * 100
            
            col1.metric("Total Executed Trades", len(trades_df))
            col2.metric("Win Rate (%)", f"{win_rate:.1f}%")
            col3.metric("Net Strategy Points", f"{trades_df['P&L Points'].sum():+.2f}")
            col4.metric("Trading Windows Tracked", f"{len(results_df)} Candles")
            
            # Closed Trades Table Logging
            st.subheader("📋 Closed Trade Performance Sheet")
            def style_pnl(val):
                return f"color: {'green' if val > 0 else 'red'}; font-weight: bold;"
            
            st.dataframe(trades_df.style.map(style_pnl, subset=['P&L Points']))
            
            # Interactive Candlestick Chart Plotly
            st.subheader("📈 Technical Chart Overlay")
            fig = go.Figure(data=[go.Candlestick(x=results_df.index, open=results_df['open'], high=results_df['high'], low=results_df['low'], close=results_df['close'], name='Nifty 50')])
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['vwap'], line=dict(color='orange', width=1.2), name='VWAP'))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['ema_20'], line=dict(color='blue', width=1.2), name='EMA'))
            
            ce_entries = results_df[results_df['signal'] == "BUY CALL (CE)"]
            pe_entries = results_df[results_df['signal'] == "BUY PUT (PE)"]
            fig.add_trace(go.Scatter(x=ce_entries.index, y=ce_entries['close'], mode='markers', marker=dict(symbol='triangle-up', size=13, color='green'), name='CE Entry'))
            fig.add_trace(go.Scatter(x=pe_entries.index, y=pe_entries['close'], mode='markers', marker=dict(symbol='triangle-down', size=13, color='red'), name='PE Entry'))
            
            fig.update_layout(xaxis_rangeslider_visible=False, height=550)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No trades found matching these constraints. Lower your volume slider or increase your tracking duration.")
    else:
        st.info("No data fetched. Ensure your date boundaries fall within the maximum limit (Yahoo Finance provides 5-min intervals within the trailing 60 days).")
