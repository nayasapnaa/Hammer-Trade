import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta, time

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="Nifty Live Backtester v2.0", layout="wide")
st.title("📊 Nifty 50 Live Historical Strategy Backtester v2.0")
st.write("Evaluate upgraded intraday 5-minute candle trading strategies with VWAP confluence, ATR buffers, and execution time windows.")

# ==========================================
# SIDEBAR: CONTROLS & STRATEGY PARAMETERS
# ==========================================
st.sidebar.header("1. Historical Data Window")
max_date = datetime.today().date()
min_date = max_date - timedelta(days=20)

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=max_date - timedelta(days=59), max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=max_date - timedelta(days=59), max_value=max_date)

st.sidebar.header("2. Strategy Configurations")
trade_setup = st.sidebar.selectbox("Candlestick Trigger Style", ["All Patterns (High Frequency)", "Only Hammers/Stars", "Only Engulfing"])
risk_reward_ratio = st.sidebar.slider("Risk-to-Reward Ratio (RRR)", 1.0, 4.0, 1.5, 0.5)
ema_length = st.sidebar.number_input("EMA Length", min_value=5, max_value=50, value=20)

st.sidebar.header("3. Enhanced Risk & Trend Filters")
atr_sl_mult = st.sidebar.slider("ATR Stop-Loss Buffer Multiplier", 0.0, 1.0, 0.2, 0.1, help="Adds ATR buffer to SL to reduce market noise stop-outs")
require_vwap = st.sidebar.checkbox("Require VWAP Trend Confluence", value=True, help="CE requires close > VWAP; PE requires close < VWAP")
apply_time_window = st.sidebar.checkbox("Restrict Execution Window (09:30 - 14:45)", value=True, help="Avoids opening gap spikes and low-volume market close")
pattern_strictness = st.sidebar.selectbox("Candle Shadow Strictness", ["Stricter Shadows (>= 1.5x Body)", "Standard Shadows (>= 1.1x Body)"])

# ==========================================
# BACKTEST CORE ENGINE
# ==========================================
def process_data(raw_df, rrr, ema_len, setup_mode, atr_mult, vwap_filter, time_filter, strict_geom):
    # Safely unpack yfinance MultiIndex or single-level columns
    flat_data = {}
    for col in raw_df.columns:
        col_name = col[0] if isinstance(col, tuple) else col
        name = str(col_name).lower().strip()
        if name in ['open', 'high', 'low', 'close', 'volume']:
            flat_data[name] = raw_df[col].to_numpy().flatten().astype(float)

    df = pd.DataFrame(flat_data, index=raw_df.index).dropna()

    if df.empty or len(df) < 10:
        return df, pd.DataFrame()

    # Align Datetime Index to Indian Standard Time (Asia/Kolkata)
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Kolkata')

    open_arr = df['open'].to_numpy()
    high_arr = df['high'].to_numpy()
    low_arr = df['low'].to_numpy()
    close_arr = df['close'].to_numpy()
    volume_arr = df['volume'].to_numpy()

    # Technical Indicators
    df['ema_20'] = df['close'].ewm(span=ema_len, adjust=False).mean()
    df['date_group'] = df.index.date

    # Intraday VWAP Calculation
    typical_price = (high_arr + low_arr + close_arr) / 3.0
    df['tp_vol'] = typical_price * volume_arr
    df['cum_tp_vol'] = df.groupby('date_group')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('date_group')['volume'].cumsum()
    cum_vol_arr = df['cum_vol'].to_numpy()
    df['vwap'] = np.where(cum_vol_arr > 0, df['cum_tp_vol'] / cum_vol_arr, close_arr)

    # 14-period ATR Calculation
    high_low = high_arr - low_arr
    high_close = np.abs(high_arr - np.roll(close_arr, 1))
    low_close = np.abs(low_arr - np.roll(close_arr, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = pd.Series(tr, index=df.index).rolling(14).mean().fillna(5.0).to_numpy()

    # Candlestick Pattern Definitions
    body = np.abs(close_arr - open_arr)
    body_eps = np.where(body == 0, 0.001, body)

    prev_open = np.zeros_like(open_arr)
    prev_close = np.zeros_like(close_arr)
    prev_open[1:] = open_arr[:-1]
    prev_close[1:] = close_arr[:-1]

    bull_eng = (close_arr > open_arr) & (prev_close < prev_open) & (open_arr <= prev_close) & (close_arr >= prev_open)
    bear_eng = (close_arr < open_arr) & (prev_close > prev_open) & (open_arr >= prev_close) & (close_arr <= prev_open)

    lower_shadow = np.where(close_arr > open_arr, open_arr - low_arr, close_arr - low_arr)
    upper_shadow = np.where(close_arr > open_arr, high_arr - close_arr, high_arr - open_arr)

    shadow_ratio = 1.5 if "Stricter" in str(strict_geom) else 1.1
    upper_max_ratio = 0.5 if "Stricter" in str(strict_geom) else 0.6

    bull_hammer = (lower_shadow >= (shadow_ratio * body_eps)) & (upper_shadow <= (upper_max_ratio * body_eps))
    bear_star = (upper_shadow >= (shadow_ratio * body_eps)) & (lower_shadow <= (upper_max_ratio * body_eps))

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

    t_start = time(9, 30)
    t_end = time(14, 45)

    for i in range(1, len(df)):
        current_time = df.index[i]
        current_close = float(close_arr[i])
        current_high = float(high_arr[i])
        current_low = float(low_arr[i])
        current_time_only = current_time.time()
        
        if in_position:
            candle_count += 1
            if pos_type == "CE":
                if current_low <= sl_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)",
                                      "Entry": entry_price, "Exit": sl_price, "P&L Points": round(sl_price - entry_price, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                elif current_high >= tp_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)",
                                      "Entry": entry_price, "Exit": tp_price, "P&L Points": round(tp_price - entry_price, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                elif candle_count >= 5:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "CALL (CE)",
                                      "Entry": entry_price, "Exit": current_close, "P&L Points": round(current_close - entry_price, 2), "Outcome": "TIME OUT"})
                    in_position = False
            elif pos_type == "PE":
                if current_high >= sl_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)",
                                      "Entry": entry_price, "Exit": sl_price, "P&L Points": round(entry_price - sl_price, 2), "Outcome": "STOP LOSS"})
                    in_position = False
                elif current_low <= tp_price:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)",
                                      "Entry": entry_price, "Exit": tp_price, "P&L Points": round(entry_price - tp_price, 2), "Outcome": "TARGET HIT"})
                    in_position = False
                elif candle_count >= 5:
                    trade_log.append({"Entry Time": entry_time, "Exit Time": current_time, "Type": "PUT (PE)",
                                      "Entry": entry_price, "Exit": current_close, "P&L Points": round(entry_price - current_close, 2), "Outcome": "TIME OUT"})
                    in_position = False
            continue

        current_ema = float(df.iloc[i]['ema_20'])
        current_vwap = float(df.iloc[i]['vwap'])
        current_atr = float(df.iloc[i]['atr'])

        in_time_win = (current_time_only >= t_start and current_time_only <= t_end) if time_filter else True
        vwap_ok_ce = (current_close > current_vwap) if vwap_filter else True
        vwap_ok_pe = (current_close < current_vwap) if vwap_filter else True

        # CALL Entry Condition
        if bool(df.iloc[i]['is_bull']) and (current_close > current_ema) and vwap_ok_ce and in_time_win:
            df.at[current_time, 'signal'] = "BUY CALL (CE)"
            in_position, pos_type, entry_price = True, "CE", current_close
            sl_price = current_low - (atr_mult * current_atr)
            tp_price = entry_price + ((entry_price - sl_price) * rrr)
            entry_time, candle_count = current_time, 0

        # PUT Entry Condition
        elif bool(df.iloc[i]['is_bear']) and (current_close < current_ema) and vwap_ok_pe and in_time_win:
            df.at[current_time, 'signal'] = "BUY PUT (PE)"
            in_position, pos_type, entry_price = True, "PE", current_close
            sl_price = current_high + (atr_mult * current_atr)
            tp_price = entry_price - ((sl_price - entry_price) * rrr)
            entry_time, candle_count = current_time, 0

    return df, pd.DataFrame(trade_log)

# ==========================================
# LIVE DATA FETCH & APP DISPLAY
# ==========================================
if start_date > end_date:
    st.error("Error: Start Date must be prior to End Date.")
else:
    ticker_symbol = "^NSEI"
    with st.spinner("Connecting to exchanges to download live Nifty data..."):
        raw_data = yf.download(tickers=ticker_symbol, start=start_date, end=end_date + timedelta(days=1), interval="5m")

    if not raw_data.empty:
        results_df, trades_df = process_data(
            raw_data.copy(), risk_reward_ratio, ema_length, trade_setup,
            atr_sl_mult, require_vwap, apply_time_window, pattern_strictness
        )

        if not trades_df.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            wins = len(trades_df[trades_df['P&L Points'] > 0])
            win_rate = (wins / len(trades_df)) * 100
            gross_profit = trades_df[trades_df['P&L Points'] > 0]['P&L Points'].sum()
            gross_loss = abs(trades_df[trades_df['P&L Points'] < 0]['P&L Points'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

            col1.metric("Total Executed Trades", len(trades_df))
            col2.metric("Win Rate (%)", f"{win_rate:.1f}%")
            col3.metric("Net Strategy Points", f"{trades_df['P&L Points'].sum():+.2f}")
            col4.metric("Profit Factor", f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A")
            col5.metric("Candles Processed", f"{len(results_df)}")

            st.subheader("📋 Closed Trade Performance Sheet")
            def style_pnl(val):
                return f"color: {'green' if val > 0 else 'red'}; font-weight: bold;"

            st.dataframe(trades_df.style.map(style_pnl, subset=['P&L Points']), use_container_width=True)

            st.subheader("📈 Interactive Technical Chart Overlay")
            fig = go.Figure(data=[go.Candlestick(
                x=results_df.index, open=results_df['open'], high=results_df['high'],
                low=results_df['low'], close=results_df['close'], name='Nifty 50'
            )])
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['vwap'], line=dict(color='orange', width=1.2), name='VWAP'))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['ema_20'], line=dict(color='blue', width=1.2), name=f'EMA ({ema_length})'))

            # Signal Markers
            buy_ce = results_df[results_df['signal'] == "BUY CALL (CE)"]
            buy_pe = results_df[results_df['signal'] == "BUY PUT (PE)"]

            fig.add_trace(go.Scatter(x=buy_ce.index, y=buy_ce['low'] - 10, mode='markers',
                                     marker=dict(symbol='triangle-up', size=12, color='green'), name='BUY CALL'))
            fig.add_trace(go.Scatter(x=buy_pe.index, y=buy_pe['high'] + 10, mode='markers',
                                     marker=dict(symbol='triangle-down', size=12, color='red'), name='BUY PUT'))

            fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"⚠️ Data parsed ({len(results_df)} candles), but no trades triggered. Try adjusting strategy filters or expanding date boundaries.")
    else:
        st.info("No data fetched. Ensure your date boundaries fall within the trailing 60 days.")
