
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# ==============================================================
# 📊 PAGE SETUP
# ==============================================================
st.set_page_config(page_title="Nifty Live Backtester", layout="wide")
st.title("📊 Nifty 50 Live Historical Strategy Tester")
st.write("Select a custom historical time window from the calendar to instantly fetch data and run the backtester.")

# ==============================================================
# 🎛️ SIDEBAR: DYNAMIC DATE & STRATEGY CONTROLS
# ==============================================================
st.sidebar.header("1. Choose Time Duration")
max_date = datetime.today().date()
min_date = max_date - timedelta(days=20)

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=max_date - timedelta(days=59), max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=max_date - timedelta(days=59), max_value=max_date)

st.sidebar.header("2. Strategy Configurations")
trade_setup = st.sidebar.selectbox("Candlestick Trigger Style", ["All Patterns (High Frequency)", "Only Hammers/Stars", "Only Engulfing"])
risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio (RRR)", 1.0, 4.0, 1.5, 0.5)
ema_length = st.sidebar.number_input("EMA Length", min_value=5, max_value=50, value=20)

# ==============================================================
# ⚙️ BACKTEST CORE ENGINE
# ==============================================================
def process_data(raw_df, rrr, ema_len, setup_mode):
    # Extract name layer whether column is a nested multi-index tuple or standard string
    flat_data = {}
    for col in raw_df.columns:
        name = col.lower().strip() if isinstance(col, tuple) else str(col).lower().strip()
        if name in ['open', 'high', 'low', 'close', 'volume']:
            flat_data[name] = raw_df[col].to_numpy().flatten().astype(float)

    # Reassemble a completely standard flat single-level column DataFrame
    df = pd.DataFrame(flat_data, index=raw_df.index).dropna()

    if df.empty or len(df) < 10:
        return df, pd.DataFrame()

    # Extract clean 1D arrays for pure math operations
    open_arr = df['open'].to_numpy()
    high_arr = df['high'].to_numpy()
    low_arr = df['low'].to_numpy()
    close_arr = df['close'].to_numpy()
    volume_arr = df['volume'].to_numpy()

    # Technical Indicators Calculations (Calculated on standardized flat float fields)
    df['ema_20'] = df['close'].ewm(span=ema_len, adjust=False).mean()
    df['date_group'] = df.index.date

    typical_price = (high_arr + low_arr + close_arr) / 3.0
    df['tp_vol'] = typical_price * volume_arr
    df['cum_tp_vol'] = df.groupby('date_group')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('date_group')['volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']

    # Core Pattern Geometry Formulas
    body = np.abs(close_arr - open_arr)
    candle_range = high_arr - low_arr
    body_eps = np.where(body == 0, 0.001, body)  # Prevent division by zero errors

    prev_open = np.zeros_like(open_arr)
    prev_close = np.zeros_like(close_arr)
    prev_open[1:] = open_arr[:-1]
    prev_close[1:] = close_arr[:-1]

    # Mathematical definitions for high frequency candle triggers
    bull_eng = (close_arr > open_arr) & (prev_close < prev_open) & (open_arr <= prev_close) & (close_arr >= prev_open)
    bear_eng = (close_arr < open_arr) & (prev_close > prev_open) & (open_arr >= prev_close) & (close_arr <= prev_open)

    lower_shadow = np.where(close_arr > open_arr, open_arr - low_arr, close_arr - low_arr)
    upper_shadow = np.where(close_arr > open_arr, high_arr - close_arr, high_arr - open_arr)

    bull_hammer = (lower_shadow > (1.1 * body_eps)) & (upper_shadow < (0.6 * body_eps))
    bear_star = (upper_shadow > (1.1 * body_eps)) & (lower_shadow < (0.6 * body_eps))

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

    # Optimized Strategy Trade Simulation Loop
    for i in range(1, len(df)):
        current_time = df.index[i]
        current_close = float(df.iloc[i]['close'])
        
        if in_position:
            candle_count += 1
            if pos_type == "CE":
                if float(df.iloc[i]['low']) <= sl_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": sl_price, "P&L Points": round(sl_price - entry_price, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                elif float(df.iloc[i]['high']) >= tp_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": tp_price, "P&L Points": round(tp_price - entry_price, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                elif candle_count >= 5:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": current_close, "P&L Points": round(current_close - entry_price, 2), "Outcome": "TIME OUT"})
                    in_position = False
            elif pos_type == "PE":
                if float(df.iloc[i]['high']) >= sl_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": sl_price, "P&L Points": round(entry_price - sl_price, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                elif float(df.iloc[i]['low']) <= tp_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": tp_price, "P&L Points": round(entry_price - tp_price, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                elif candle_count >= 5:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": current_close, "P&L Points": round(entry_price - current_close, 2), "Outcome": "TIME OUT"})
                    in_position = False
            continue
            
        current_ema = float(df.iloc[i]['ema_20'])
        
        # Valid entry matching using clean floating-point numerical constraints
        if bool(df.iloc[i]['is_bull']) and (current_close > current_ema):
            df.iloc[i, df.columns.get_loc('signal')] = "BUY CALL (CE)"
            in_position, pos_type, entry_price = True, "CE", current_close
            sl_price = float(df.iloc[i]['low'])
            tp_price = entry_price + ((entry_price - sl_price) * rrr)
            entry_time, candle_count = current_time, 0
        elif bool(df.iloc[i]['is_bear']) and (current_close < current_ema):
            df.iloc[i, df.columns.get_loc('signal')] = "BUY PUT (PE)"
            in_position, pos_type, entry_price = True, "PE", current_close
            sl_price = float(df.iloc[i]['high'])
            tp_price = entry_price - ((sl_price - entry_price) * rrr)
            entry_time, candle_count = current_time, 0
            
    return df, pd.DataFrame(trade_log)

# ==============================================================
# 🚀 LIVE API DATA FETCH TRIGGER
# ==============================================================
if start_date > end_date:
    st.error("Error: Start Date must be prior to End Date.")
else:
    ticker_symbol = "^NSEI"
    with st.spinner("Connecting to live exchanges to download Nifty data..."):
        raw_data = yf.download(tickers=ticker_symbol, start=start_date, end=end_date + timedelta(days=1), interval="5m")
        
    if not raw_data.empty:
        results_df, trades_df = process_data(raw_data.copy(), risk_reward_ratio, ema_length, trade_setup)
        
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
            st.warning(f"⚠️ Data parsed ({len(results_df)} candles), but no trades triggered. Try widening your date duration to scan a larger market window.")
    else:
        st.info("No data fetched. Ensure your date boundaries fall within the trailing 60 days.")
