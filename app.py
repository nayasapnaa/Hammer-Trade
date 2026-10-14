import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty High-Frequency Tester", layout="wide")
st.title("📊 Nifty 50 High-Frequency Strategy Tester")
st.write("Optimized setup to capture higher intraday trade volumes using multiple candle triggers.")

# --- SIDEBAR CONTROL ---
st.sidebar.header("Strategy Configurations")
trade_setup = st.sidebar.selectbox("Candlestick Trigger Style", ["All Patterns (High Frequency)", "Only Hammers/Stars", "Only Engulfing"])
risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio (RRR)", 1.0, 4.0, 1.5, 0.5)
ema_length = st.sidebar.number_input("EMA Length", min_value=5, max_value=50, value=20)
vol_multiplier = st.sidebar.slider("Volume Filter (Lower = More Trades)", 0.5, 2.0, 0.8, 0.1)

# --- ADVANCED HIGH FREQUENCY ENGINE ---
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
    
    # Candle Dimensions for Math Parsing
    body = (df['close'] - df['open']).abs()
    candle_range = df['high'] - df['low']
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    # 1. Engulfing Definitions
    bull_eng = (df['close'] > df['open']) & (df['prev_close'] < df['prev_open']) & (df['open'] <= df['prev_close']) & (df['close'] >= df['prev_open'])
    bear_eng = (df['close'] < df['open']) & (df['prev_close'] > df['prev_open']) & (df['open'] >= df['prev_close']) & (df['close'] <= df['prev_open'])
    
    # 2. Hammer / Shooting Star Definitions (Lower shadow or Upper shadow dominates)
    lower_shadow = np.where(df['close'] > df['open'], df['open'] - df['low'], df['close'] - df['low'])
    upper_shadow = np.where(df['close'] > df['open'], df['high'] - df['close'], df['high'] - df['open'])
    
    bull_hammer = (lower_shadow > (2 * body)) & (upper_shadow < (0.5 * body)) & (candle_range > 0)
    bear_star = (upper_shadow > (2 * body)) & (lower_shadow < (0.5 * body)) & (candle_range > 0)
    
    # Filter selection based on UI input
    if setup_mode == "Only Engulfing":
        df['is_bull'] = bull_eng
        df['is_bear'] = bear_eng
    elif setup_mode == "Only Hammers/Stars":
        df['is_bull'] = bull_hammer
        df['is_bear'] = bear_star
    else: # All Patterns
        df['is_bull'] = bull_eng | bull_hammer
        df['is_bear'] = bear_eng | bear_star

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
            # Exit evaluation
            if pos_type == "CE":
                if df.iloc[i]['low'] <= sl_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": sl_price, "P&L Points": round(sl_price - entry_price, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                elif df.iloc[i]['high'] >= tp_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": tp_price, "P&L Points": round(tp_price - entry_price, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                elif candle_count >= 5: # Extended to 5 candles (25 mins) to give trades room
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
            
        # Entry logic
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

# --- UI DATA RENDER ---
uploaded_file = st.file_uploader("Upload Nifty 50 CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    date_col = [c for c in data.columns if 'date' in c.lower() or 'time' in c.lower()][0]
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    data.sort_index(inplace=True)
    
    results_df, trades_df = process_data(data.copy(), risk_reward_ratio, ema_length, vol_multiplier, trade_setup)
    
    if not trades_df.empty:
        col1, col2, col3 = st.columns(3)
        win_rate = (len(trades_df[trades_df['P&L Points'] > 0]) / len(trades_df)) * 100
        col1.metric("Total Executed Trades", len(trades_df))
        col2.metric("Win Rate (%)", f"{win_rate:.1f}%")
        col3.metric("Net Points", f"{trades_df['P&L Points'].sum():+.2f}")
        
        st.subheader("📋 Closed Trade Log")
        def style_pnl(val):
            return f"color: {'green' if val > 0 else 'red'}; font-weight: bold;"
        
        # FIXED: Using modern .map() instead of .applymap() to avoid the AttributeError
        st.dataframe(trades_df.style.map(style_pnl, subset=['P&L Points']))
    else:
        st.warning("No trades found. Lower your Volume Filter in the sidebar.")
