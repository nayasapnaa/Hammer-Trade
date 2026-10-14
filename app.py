import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty 50 Strategy Tester", layout="wide")
st.title("📊 Nifty 50 5-Min Options Strategy Tester")
st.write("Upload your 5-minute historical index data, adjust settings, and view exact Profit & Loss outputs instantly.")

# --- SIDEBAR INTERFACE CONTROL ---
st.sidebar.header("Strategy Configurations")
risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio (RRR)", 1.0, 4.0, 2.0, 0.5)
vol_multiplier = st.sidebar.slider("Volume Spike Multiplier", 1.0, 3.0, 1.5, 0.1)
ema_length = st.sidebar.number_input("EMA Length", min_value=5, max_value=50, value=20)

# --- ADVANCED TRADE SIMULATOR WITH PNL ---
def process_data(df, rrr, vol_mult, ema_len):
    df.columns = [col.lower() for col in df.columns]
    
    # 1. Technical Indicators (Pure Pandas)
    df['ema_20'] = df['close'].ewm(span=ema_len, adjust=False).mean()
    df['date_group'] = df.index.date
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    tp_vol = typical_price * df['volume']
    df['cum_tp_vol'] = tp_vol.groupby(df['date_group']).cumsum()
    df['cum_vol'] = df['volume'].groupby(df['date_group']).cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    
    df['vol_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_spike'] = df['volume'] > (df['vol_sma'] * vol_mult)
    
    # 2. Candle Patterns
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['bullish_engulfing'] = (df['close'] > df['open']) & (df['prev_close'] < df['prev_open']) & (df['open'] <= df['prev_close']) & (df['close'] >= df['prev_open'])
    df['bearish_engulfing'] = (df['close'] < df['open']) & (df['prev_close'] > df['prev_open']) & (df['open'] >= df['prev_close']) & (df['close'] <= df['prev_open'])
    
    # 3. Initialize Arrays for Strategy Tracking
    trade_log = []
    in_position = False
    pos_type = None  # "CE" or "PE"
    entry_price = 0
    sl_price = 0
    tp_price = 0
    entry_time = None
    candle_count = 0
    
    # Setup columns for visualization mapping
    df['signal'] = "HOLD"
    
    for i in range(1, len(df)):
        current_time = df.index[i]
        current_open = df.iloc[i]['open']
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        current_close = df.iloc[i]['close']
        
        # --- IF WE ARE ALREADY IN A TRADE: TRACK EXIT CONDITIONS ---
        if in_position:
            candle_count += 1
            
            if pos_type == "CE":
                # Check Stop Loss
                if current_low <= sl_price:
                    pnl = sl_price - entry_price
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": sl_price, "P&L Points": round(pnl, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                # Check Target
                elif current_high >= tp_price:
                    pnl = tp_price - entry_price
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": tp_price, "P&L Points": round(pnl, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                # Check 15-Min Time limit (3 candles)
                elif candle_count >= 3:
                    pnl = current_close - entry_price
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)", "Entry": entry_price, "Exit": current_close, "P&L Points": round(pnl, 2), "Outcome": "TIME OUT"})
                    in_position = False
                    
            elif pos_type == "PE":
                # Check Stop Loss
                if current_high >= sl_price:
                    pnl = entry_price - sl_price
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": sl_price, "P&L Points": round(pnl, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                # Check Target
                elif current_low <= tp_price:
                    pnl = entry_price - tp_price
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": tp_price, "P&L Points": round(pnl, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                # Check 15-Min Time limit (3 candles)
                elif candle_count >= 3:
                    pnl = entry_price - current_close
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)", "Entry": entry_price, "Exit": current_close, "P&L Points": round(pnl, 2), "Outcome": "TIME OUT"})
                    in_position = False
            continue
            
        # --- SCAN FOR NEW SIGNALS (ONLY IF SIDEWAYS/FLAT TO START NEW POSITION) ---
        current_ema = df.iloc[i]['ema_20']
        current_vwap = df.iloc[i]['vwap']
        
        # Bullish Call Setup
        if (current_close > current_ema) and (current_close > current_vwap) and df.iloc[i]['bullish_engulfing'] and df.iloc[i]['volume_spike']:
            df.iloc[i, df.columns.get_loc('signal')] = "BUY CALL (CE)"
            in_position = True
            pos_type = "CE"
            entry_price = current_close
            sl_price = df.iloc[i]['low']
            risk = entry_price - sl_price
            tp_price = entry_price + (risk * rrr)
            entry_time = current_time
            candle_count = 0
            
        # Bearish Put Setup
        elif (current_close < current_ema) and (current_close < current_vwap) and df.iloc[i]['bearish_engulfing'] and df.iloc[i]['volume_spike']:
            df.iloc[i, df.columns.get_loc('signal')] = "BUY PUT (PE)"
            in_position = True
            pos_type = "PE"
            entry_price = current_close
            sl_price = df.iloc[i]['high']
            risk = sl_price - entry_price
            tp_price = entry_price - (risk * rrr)
            entry_time = current_time
            candle_count = 0
            
    return df, pd.DataFrame(trade_log)

# --- SIMULATOR / DATA LOADER ---
uploaded_file = st.file_uploader("Upload Nifty 50 5-Min CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    date_col = [c for c in data.columns if 'date' in c.lower() or 'time' in c.lower()][0]
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    data.sort_index(inplace=True)
    
    # Process Strategy Engines
    with st.spinner("Simulating historical execution tracks..."):
        results_df, trades_df = process_data(data.copy(), risk_reward_ratio, vol_multiplier, ema_length)
    
    # --- PERFORMANCE KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    
    if not trades_df.empty:
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['P&L Points'] > 0])
        win_rate = (profitable_trades / total_trades) * 100
        net_points = trades_df['P&L Points'].sum()
        
        with col1:
            st.metric("Total Executed Trades", total_trades)
        with col2:
            st.metric("Win Rate (%)", f"{win_rate:.1f}%")
        with col3:
            st.metric("Net Strategy Points", f"{net_points:+.2f}")
        with col4:
            st.metric("Profitable vs Losing", f"{profitable_trades}W - {total_trades - profitable_trades}L")
            
        # --- LOG DISPLAY PANEL ---
        st.subheader("📋 Complete Profit & Loss Closed Trade Log")
        
        # Color coding rows for readability
        def style_pnl(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(trades_df.style.applymap(style_pnl, subset=['P&L Points']))
        
        # --- GRAPH VISUALIZATION ---
        st.subheader("📈 Chart Visualization")
        fig = go.Figure(data=[go.Candlestick(x=results_df.index, open=results_df['open'], high=results_df['high'], low=results_df['low'], close=results_df['close'], name='Nifty 50')])
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['vwap'], line=dict(color='orange', width=1.2), name='VWAP'))
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['ema_20'], line=dict(color='blue', width=1.2), name='EMA'))
        
        # Dynamic Signals Overlay mapping
        ce_entries = results_df[results_df['signal'] == "BUY CALL (CE)"]
        pe_entries = results_df[results_df['signal'] == "BUY PUT (PE)"]
        fig.add_trace(go.Scatter(x=ce_entries.index, y=ce_entries['close'], mode='markers', marker=dict(symbol='triangle-up', size=13, color='green'), name='CE Buy'))
        fig.add_trace(go.Scatter(x=pe_entries.index, y=pe_entries['close'], mode='markers', marker=dict(symbol='triangle-down', size=13, color='red'), name='PE Buy'))
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=550)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ No trades matched your strategy constraints with the current settings. Try lowering the Volume Spike Multiplier.")
else:
    st.info("💡 Awaiting CSV historical template data to generate metrics panel metrics.")
