import os, warnings, asyncio, aiohttp
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import numpy as np
import pandas as pd

import yfinance as yf
from fastapi import FastAPI
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from joblib import dump

warnings.filterwarnings("ignore")

# ---- TELEGRAM CREDENTIALS (embedded) ----
TELEGRAM_BOT_TOKEN = "8420126239:AAG7NuXxk4uM9Izjh1xzw3jbfhwu75vd7QM"
TELEGRAM_CHAT_ID = "5957769856"

# ---- CONFIG ----
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","GC=F,BTC-USD,ETH-USD").split(",") if s.strip()]
INTERVAL = "15m"
LOOKBACK_DAYS = 60
SCHEDULE_MINUTES = 3

app = FastAPI(title="Sniper AI Bot")

MODELS: Dict[str, RandomForestClassifier] = {}
FEATURE_COLUMNS: List[str] = []
LAST_SIGNAL_TS: Dict[str, pd.Timestamp] = {}

def fetch_ohlc(symbol: str, period_days: int, interval: str) -> pd.DataFrame:
    period = f"{period_days}d"
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={c:c.capitalize() for c in df.columns})
    df = df.dropna().copy()
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = ta.rsi(out["Close"], length=14)
    out["ema_fast"] = ta.ema(out["Close"], length=9)
    out["ema_slow"] = ta.ema(out["Close"], length=21)
    macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
    out["macd"] = macd["MACD_12_26_9"]
    out["macd_sig"] = macd["MACDs_12_26_9"]
    out["macd_hist"] = macd["MACDh_12_26_9"]
    out["obv"] = ta.obv(out["Close"], out["Volume"])
    out["atr"] = ta.atr(out["High"], out["Low"], out["Close"], length=14)
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_6"] = out["Close"].pct_change(6)
    out["vol_ma"] = ta.sma(out["Volume"], length=20)
    out["vol_spike"] = (out["Volume"] > 1.5 * out["vol_ma"]).astype(int)
    out = out.dropna().copy()
    return out

def sniper_rule_signal(row) -> int:
    buy = (row["rsi"] < 30) and (row["ema_fast"] > row["ema_slow"]) and ((row["macd_hist"] > 0) or (row["vol_spike"] == 1))
    sell = (row["rsi"] > 70) and (row["ema_fast"] < row["ema_slow"]) and ((row["macd_hist"] < 0) or (row["vol_spike"] == 1))
    if buy and not sell:
        return 1
    if sell and not buy:
        return -1
    return 0

def label_future(df: pd.DataFrame, horizon: int = 3, threshold_bp: float = 0.0) -> pd.Series:
    fut_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    labels = np.where(fut_ret > threshold_bp/10000.0, 1, np.where(fut_ret < -threshold_bp/10000.0, -1, 0))
    return pd.Series(labels, index=df.index)

def build_dataset(symbol: str, days: int, interval: str, horizon: int = 3) -> pd.DataFrame:
    df = fetch_ohlc(symbol, days, interval)
    if df.empty:
        return df
    df = add_features(df)
    df["y"] = label_future(df, horizon=horizon, threshold_bp=0.0)
    return df.dropna()

def fit_model(symbol: str, df: pd.DataFrame) -> Dict[str, float]:
    global FEATURE_COLUMNS
    X = df[["rsi","ema_fast","ema_slow","macd","macd_sig","macd_hist","obv","atr","ret_1","ret_3","ret_6","vol_spike"]].copy()
    y = df["y"].astype(int)
    FEATURE_COLUMNS = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced_subsample")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    MODELS[symbol] = clf
    try:
        os.makedirs("models", exist_ok=True)
        dump(clf, f"models/{symbol.replace('/','_')}_rf.joblib")
    except Exception:
        pass
    return {"accuracy": acc, "precision": prec, "recall": rec}

def model_predict(symbol: str, row: pd.Series) -> Dict[str, float]:
    clf = MODELS.get(symbol)
    if clf is None:
        return {"pred": 0, "p_up": 0.0, "p_down": 0.0}
    X = row[FEATURE_COLUMNS].values.reshape(1, -1)
    proba = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None
    if proba is not None and len(proba) == 3:
        cls_order = list(clf.classes_)
        def p_of(c):
            return float(proba[cls_order.index(c)]) if c in cls_order else 0.0
        return {"pred": int(clf.predict(X)[0]), "p_up": p_of(1), "p_down": p_of(-1)}
    pred = int(clf.predict(X)[0])
    return {"pred": pred, "p_up": 0.5 if pred==1 else 0.0, "p_down": 0.5 if pred==-1 else 0.0}

async def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": str(TELEGRAM_CHAT_ID), "text": text, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=20) as resp:
            await resp.text()

def sl_tp_from_atr(price: float, atr: float, mul_sl=1.5, mul_tp=2.5):
    sl = atr * mul_sl
    tp = atr * mul_tp
    return sl, tp

class AnalyzeOut(BaseModel):
    symbol: str
    time: str
    price: float
    rule_signal: int
    ml_pred: int
    p_up: float
    p_down: float
    decision: str
    sl: float
    tp1: float
    tp2: float

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/price")
def price(symbol: str):
    df = fetch_ohlc(symbol, 2, INTERVAL)
    if df.empty:
        return {"symbol": symbol, "price": None, "time": None}
    px = float(df["Close"].iloc[-1])
    return {"symbol": symbol, "price": px, "time": df.index[-1].isoformat()}

@app.post("/train")
def train(symbol: str, days: int = LOOKBACK_DAYS, interval: str = INTERVAL):
    df = build_dataset(symbol, days, interval)
    if df.empty:
        return {"error":"no data","symbol":symbol}
    metrics = fit_model(symbol, df)
    return {"symbol": symbol, "rows": int(len(df)), "interval": interval, "metrics": metrics}

@app.get("/backtest")
def backtest(symbol: str, days: int = LOOKBACK_DAYS, interval: str = INTERVAL, use_ml: bool = True):
    df = build_dataset(symbol, days, interval)
    if df.empty:
        return {"error":"no data","symbol":symbol}
    signals_rule = df.apply(sniper_rule_signal, axis=1)
    if use_ml and symbol in MODELS:
        preds = []
        for _, row in df.iterrows():
            preds.append(model_predict(symbol, row)["pred"])
        signals_ml = pd.Series(preds, index=df.index)
        signals = np.where(signals_rule!=0, signals_rule, signals_ml)
    else:
        signals = signals_rule.values
    horizon = 6
    pnl = []
    in_pos = 0
    entry_px = 0.0
    bars_held = 0
    for i in range(len(df)-1):
        sig = int(signals[i])
        close_i = float(df["Close"].iloc[i+1])
        if in_pos == 0 and sig != 0:
            in_pos = sig
            entry_px = close_i
            bars_held = 0
        elif in_pos != 0:
            bars_held += 1
            if (sig == -in_pos and sig != 0) or (bars_held >= horizon):
                ret = (close_i/entry_px - 1.0) * in_pos
                pnl.append(ret)
                in_pos = 0
    total_ret = float(np.sum(pnl)) if pnl else 0.0
    win_rate = float(np.mean([1 if r>0 else 0 for r in pnl])) if pnl else 0.0
    trades = int(len(pnl))
    return {"symbol": symbol, "interval": interval, "days": days, "trades": trades, "win_rate": round(win_rate,3), "total_return": round(total_ret,4)}

@app.get("/analyze", response_model=AnalyzeOut)
def analyze(symbol: str):
    df = fetch_ohlc(symbol, 5, INTERVAL)
    if df.empty:
        return AnalyzeOut(symbol=symbol, time="", price=0.0, rule_signal=0, ml_pred=0, p_up=0.0, p_down=0.0, decision="HOLD", sl=0.0, tp1=0.0, tp2=0.0)
    df = add_features(df)
    row = df.iloc[-1]
    rule_sig = sniper_rule_signal(row)
    ml = model_predict(symbol, row)
    decision = "HOLD"
    if rule_sig == 1 and (ml["pred"]==1 or ml["p_up"]>=0.55):
        decision = "BUY"
    elif rule_sig == -1 and (ml["pred"]==-1 or ml["p_down"]>=0.55):
        decision = "SELL"
    atr = float(row["atr"])
    px = float(row["Close"])
    sl_raw, tp_raw = sl_tp_from_atr(px, atr)
    if decision == "BUY":
        sl = px - sl_raw
        tp1 = px + tp_raw
        tp2 = px + 1.5*tp_raw
    elif decision == "SELL":
        sl = px + sl_raw
        tp1 = px - tp_raw
        tp2 = px - 1.5*tp_raw
    else:
        sl, tp1, tp2 = 0.0, 0.0, 0.0
    return AnalyzeOut(symbol=symbol, time=df.index[-1].isoformat(), price=px, rule_signal=rule_sig, ml_pred=ml["pred"], p_up=round(ml["p_up"],3), p_down=round(ml["p_down"],3), decision=decision, sl=round(sl,2), tp1=round(tp1,2), tp2=round(tp2,2))


def job_scan_and_alert():
    """
    Clean, well-formed scan loop:
    - for each symbol ensure model exists
    - call analyze() which returns dict or Pydantic object
    - check confirmations safely and send telegram only if gated
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sym in SYMBOLS:
        try:
            # ensure model available
            if sym not in MODELS:
                df = build_dataset(sym, min(LOOKBACK_DAYS, 30), INTERVAL)
                if len(df) > 200:
                    fit_model(sym, df)

            res = analyze(sym)
            # normalize res to dict
            if hasattr(res, "__dict__"):
                res = res.__dict__
        except Exception:
            # skip symbol on any analysis error
            continue

        # safe confirmations extraction
        try:
            conf = res.get("confirmations", {}) if isinstance(res, dict) else {}
            count_buy = int(conf.get("count_buy", 0))
            count_sell = int(conf.get("count_sell", 0))
        except Exception:
            count_buy = 0
            count_sell = 0

        # gated alert: require >=4 confirmations
        if (count_buy >= 4) or (count_sell >= 4):
            try:
                ts_key = pd.Timestamp(res.get("time") or datetime.utcnow().isoformat())
            except Exception:
                ts_key = pd.Timestamp(datetime.utcnow())
            last_ts = LAST_SIGNAL_TS.get(sym)
            if last_ts is None or ts_key > last_ts:
                try:
                    text = (f"*{sym}* {res.get('decision','HOLD')}\n"
                            f"Price: {float(res.get('price',0)):.2f}\n"
                            f"Rule: {res.get('rule_signal',0)}  ML: {res.get('ml_pred',0)}  P(up): {res.get('p_up',0.0):.2f}  P(down): {res.get('p_down',0.0):.2f}\n"
                            f"SL: {float(res.get('sl',0)):.2f}  TP1: {float(res.get('tp1',0)):.2f}  TP2: {float(res.get('tp2',0)):.2f}\n"
                            f"{INTERVAL}  {res.get('time','')}")
                    loop.run_until_complete(send_telegram(text))
                    LAST_SIGNAL_TS[sym] = ts_key
                except Exception:
                    # if telegram fails continue
                    continue
scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
scheduler.add_job(job_scan_and_alert, "interval", minutes=SCHEDULE_MINUTES, next_run_time=datetime.utcnow()+timedelta(seconds=5))
scheduler.start()

@app.get("/")
def root():
    return {"service":"Sniper AI Bot","symbols":SYMBOLS,"interval":INTERVAL}

# --- robust fetch_ohlc replacement (appended) ---
import time
def fetch_ohlc(symbol: str, period_days: int, interval: str) -> pd.DataFrame:
    """
    Robust fetch:
    - handles yf.download returning tuples
    - flattens MultiIndex columns
    - retries and falls back to alternative symbols for gold
    """
    period = f"{period_days}d"
    attempts = 2
    # symbol fallbacks map
    fallbacks = {
        "GC=F": ["GC=F", "GLD"],
        "XAUUSD": ["GC=F", "GLD"]
    }
    try_symbols = [symbol] + fallbacks.get(symbol, [])
    for sym in try_symbols:
        for attempt in range(attempts):
            try:
                df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False)
                # sometimes yf.download returns (df, info) tuple in old versions
                if isinstance(df, tuple) and len(df) > 0:
                    df = df[0]
                if df is None:
                    df = pd.DataFrame()
                # if MultiIndex columns, flatten them
                if isinstance(getattr(df, "columns", None), pd.MultiIndex):
                    df.columns = ["_".join([str(i) for i in col]).strip() for col in df.columns.values]
                    # try to normalise names back to Open/High/Low/Close/Volume if present
                    cols_lower = [c.lower() for c in df.columns]
                    mapping = {}
                    if any("open" in c for c in cols_lower):
                        for c in df.columns:
                            lc = c.lower()
                            if "open" in lc: mapping[c] = "Open"
                            if "high" in lc: mapping[c] = "High"
                            if "low" in lc: mapping[c] = "Low"
                            if "close" in lc: mapping[c] = "Close"
                            if "volume" in lc: mapping[c] = "Volume"
                        if mapping:
                            df = df.rename(columns=mapping)
                # ensure expected columns exist and capitalise if simple names
                if not df.empty:
                    # normalise column names to capitalized standard if possible
                    cols = []
                    for c in df.columns:
                        if isinstance(c, str):
                            cols.append(c.capitalize())
                        else:
                            cols.append(str(c))
                    df.columns = cols
                # dropna and return if not empty
                if not df.empty:
                    df = df.dropna().copy()
                    # tag original symbol when fallback used
                    df.attrs["__fetched_symbol__"] = sym
                    return df
            except Exception:
                # brief backoff then retry
                time.sleep(1)
                continue
    # nothing found -> return empty DF
    return pd.DataFrame()
# --- end robust fetch_ohlc ---

@app.post("/ping_telegram")
async def ping_telegram():
    await send_telegram("✅ Sniper AI Bot connected. Alerts will appear on BUY/SELL only.")
    return {"sent": True}

@app.post("/scan_now")
def scan_now():
    job_scan_and_alert()
    return {"ok": True}

# === confirmations and sizing helpers ===
def compute_confirmations(row: pd.Series, ml_pred: Dict[str, float]) -> Dict[str, int | float | bool]:
    ema_ok_buy  = row["ema_fast"] > row["ema_slow"]
    ema_ok_sell = row["ema_fast"] < row["ema_slow"]
    rsi_buy  = row["rsi"] < 30
    rsi_sell = row["rsi"] > 70
    macd_buy  = row["macd_hist"] > 0
    macd_sell = row["macd_hist"] < 0
    vol_spike = int(row.get("vol_spike", 0)) == 1
    p_up  = float(ml_pred.get("p_up", 0.0))
    p_dn  = float(ml_pred.get("p_down", 0.0))
    ml_buy  = p_up >= 0.55
    ml_sell = p_dn >= 0.55
    conf_buy  = [ema_ok_buy,  rsi_buy,  macd_buy,  vol_spike, ml_buy]
    conf_sell = [ema_ok_sell, rsi_sell, macd_sell, vol_spike, ml_sell]
    return {
        "conf_buy": conf_buy,
        "conf_sell": conf_sell,
        "count_buy": int(sum(conf_buy)),
        "count_sell": int(sum(conf_sell)),
        "p_up": p_up,
        "p_down": p_dn,
    }

def position_size_units(entry: float, stop: float, account: float, risk_pct: float) -> float:
    # returns units of the asset to buy/sell given risk percent of account
    risk_amt = account * (risk_pct / 100.0)
    sl_dist = abs(entry - stop)
    if sl_dist <= 0:
        return 0.0
    return risk_amt / sl_dist

# replace analyze to include confirmations and gating
from fastapi import Query

@app.get("/analyze", response_model=AnalyzeOut)
def analyze(symbol: str):
    df = fetch_ohlc(symbol, 5, INTERVAL)
    if df.empty:
        return AnalyzeOut(symbol=symbol, time="", price=0.0, rule_signal=0, ml_pred=0, p_up=0.0, p_down=0.0, decision="HOLD", sl=0.0, tp1=0.0, tp2=0.0)
    df = add_features(df)
    row = df.iloc[-1]
    rule_sig = sniper_rule_signal(row)
    ml = model_predict(symbol, row)
    conf = compute_confirmations(row, ml)
    px = float(row["Close"])
    atr = float(row["atr"])
    sl_raw, tp_raw = sl_tp_from_atr(px, atr)  # ATR-based distances

    decision = "HOLD"
    # gate on 5-confirm system: require >=4 confirmations
    if conf["count_buy"] >= 4:
        decision = "BUY"
    elif conf["count_sell"] >= 4:
        decision = "SELL"

    if decision == "BUY":
        sl = px - sl_raw
        tp1 = px + tp_raw
        tp2 = px + 1.5 * tp_raw
    elif decision == "SELL":
        sl = px + sl_raw
        tp1 = px - tp_raw
        tp2 = px - 1.5 * tp_raw
    else:
        sl, tp1, tp2 = 0.0, 0.0, 0.0

    # pack fields (keep schema; extra details via headers-like keys in response)
    out = AnalyzeOut(
        symbol=symbol,
        time=df.index[-1].isoformat(),
        price=px,
        rule_signal=rule_sig,
        ml_pred=int(ml["pred"]),
        p_up=round(conf["p_up"], 3),
        p_down=round(conf["p_down"], 3),
        decision=decision,
        sl=round(sl, 2),
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
    )
    # attach debug confirmations for clients that read JSON (not enforced by Pydantic model)
    try:
        out_dict = out.__dict__
        out_dict["confirmations"] = {
            "buy_flags": conf["conf_buy"],
            "sell_flags": conf["conf_sell"],
            "count_buy": conf["count_buy"],
            "count_sell": conf["count_sell"],
        }
        return out_dict  # FastAPI will serialize dict directly
    except Exception:
        return out

# position sizing endpoint
@app.get("/size")
def size(
    symbol: str,
    entry: float = Query(...),
    sl: float = Query(..., description="stop-loss price"),
    account: float = Query(..., description="account balance in USD"),
    risk_pct: float = Query(1.0, description="risk percent of account"),
):
    units = position_size_units(entry, sl, account, risk_pct)
    return {"symbol": symbol, "entry": entry, "sl": sl, "account": account, "risk_pct": risk_pct, "units": round(units, 6)}
from fastapi import Query

def _compute_confirmations(row, ml):
    ema_ok_buy  = float(row["ema_fast"]) > float(row["ema_slow"])
    ema_ok_sell = float(row["ema_fast"]) < float(row["ema_slow"])
    rsi_buy  = float(row["rsi"]) < 30
    rsi_sell = float(row["rsi"]) > 70
    macd_buy  = float(row["macd_hist"]) > 0
    macd_sell = float(row["macd_hist"]) < 0
    vol_spike = int(row.get("vol_spike", 0)) == 1
    p_up  = float(ml.get("p_up", 0.0))
    p_dn  = float(ml.get("p_down", 0.0))
    ml_buy  = p_up >= 0.55
    ml_sell = p_dn >= 0.55
    conf_buy  = [ema_ok_buy,  rsi_buy,  macd_buy,  vol_spike, ml_buy]
    conf_sell = [ema_ok_sell, rsi_sell, macd_sell, vol_spike, ml_sell]
    return {"buy": conf_buy, "sell": conf_sell, "cnt_buy": sum(conf_buy), "cnt_sell": sum(conf_sell), "p_up": p_up, "p_down": p_dn}

@app.get("/analyze_strict")
def analyze_strict(symbol: str):
    df = fetch_ohlc(symbol, 5, INTERVAL)
    if df.empty:
        return {"symbol": symbol, "decision": "HOLD"}
    df = add_features(df)
    row = df.iloc[-1]
    rule_sig = sniper_rule_signal(row)
    ml = model_predict(symbol, row)
    c = _compute_confirmations(row, ml)
    px = float(row["Close"])
    atr = float(row["atr"])
    d = "HOLD"
    if c["cnt_buy"] >= 4:
        d = "BUY"
    elif c["cnt_sell"] >= 4:
        d = "SELL"
    sl_raw, tp_raw = sl_tp_from_atr(px, atr)
    if d == "BUY":
        sl = px - sl_raw
        tp1 = px + tp_raw
        tp2 = px + 1.5 * tp_raw
    elif d == "SELL":
        sl = px + sl_raw
        tp1 = px - tp_raw
        tp2 = px - 1.5 * tp_raw
    else:
        sl = tp1 = tp2 = 0.0
    return {
        "symbol": symbol,
        "time": df.index[-1].isoformat(),
        "price": round(px, 2),
        "rule_signal": int(rule_sig),
        "p_up": round(c["p_up"], 3),
        "p_down": round(c["p_down"], 3),
        "conf_buy": c["buy"],
        "conf_sell": c["sell"],
        "count_buy": int(c["cnt_buy"]),
        "count_sell": int(c["cnt_sell"]),
        "decision": d,
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
    }

@app.get("/size")
def size(symbol: str, entry: float = Query(...), sl: float = Query(...), account: float = Query(...), risk_pct: float = Query(1.0)):
    risk_amt = account * (risk_pct / 100.0)
    sl_dist = abs(entry - sl)
    units = 0.0 if sl_dist <= 0 else risk_amt / sl_dist
    return {"symbol": symbol, "entry": entry, "sl": sl, "account": account, "risk_pct": risk_pct, "units": round(units, 6)}

@app.post("/scan_now_strict")
def scan_now_strict():
    sent = []
    for sym in SYMBOLS:
        res = analyze_strict(sym)
        if res["decision"] in ("BUY","SELL"):
            txt = f"*{sym}* {res['decision']}\nPrice: {res['price']}\nP(up): {res['p_up']}  P(down): {res['p_down']}\nConf: buy={res['count_buy']} sell={res['count_sell']}\nSL: {res['sl']}  TP1: {res['tp1']}  TP2: {res['tp2']}\n{INTERVAL}  {res['time']}"
            try:
                asyncio.get_event_loop().run_until_complete(send_telegram(txt))
                sent.append(sym)
            except Exception:
                pass
    return {"ok": True, "alerts": sent}
from fastapi import Query

def _compute_confirmations(row, ml):
    ema_ok_buy  = float(row["ema_fast"]) > float(row["ema_slow"])
    ema_ok_sell = float(row["ema_fast"]) < float(row["ema_slow"])
    rsi_buy  = float(row["rsi"]) < 30
    rsi_sell = float(row["rsi"]) > 70
    macd_buy  = float(row["macd_hist"]) > 0
    macd_sell = float(row["macd_hist"]) < 0
    vol_spike = int(row.get("vol_spike", 0)) == 1
    p_up  = float(ml.get("p_up", 0.0))
    p_dn  = float(ml.get("p_down", 0.0))
    ml_buy  = p_up >= 0.55
    ml_sell = p_dn >= 0.55
    conf_buy  = [ema_ok_buy,  rsi_buy,  macd_buy,  vol_spike, ml_buy]
    conf_sell = [ema_ok_sell, rsi_sell, macd_sell, vol_spike, ml_sell]
    return {"buy": conf_buy, "sell": conf_sell, "cnt_buy": sum(conf_buy), "cnt_sell": sum(conf_sell), "p_up": p_up, "p_down": p_dn}

@app.get("/analyze_strict")
def analyze_strict(symbol: str):
    df = fetch_ohlc(symbol, 5, INTERVAL)
    if df.empty:
        return {"symbol": symbol, "decision": "HOLD"}
    df = add_features(df)
    row = df.iloc[-1]
    rule_sig = sniper_rule_signal(row)
    ml = model_predict(symbol, row)
    c = _compute_confirmations(row, ml)
    px = float(row["Close"])
    atr = float(row["atr"])
    d = "HOLD"
    if c["cnt_buy"] >= 4:
        d = "BUY"
    elif c["cnt_sell"] >= 4:
        d = "SELL"
    sl_raw, tp_raw = sl_tp_from_atr(px, atr)
    if d == "BUY":
        sl = px - sl_raw
        tp1 = px + tp_raw
        tp2 = px + 1.5 * tp_raw
    elif d == "SELL":
        sl = px + sl_raw
        tp1 = px - tp_raw
        tp2 = px - 1.5 * tp_raw
    else:
        sl = tp1 = tp2 = 0.0
    return {
        "symbol": symbol,
        "time": df.index[-1].isoformat(),
        "price": round(px, 2),
        "rule_signal": int(rule_sig),
        "p_up": round(c["p_up"], 3),
        "p_down": round(c["p_down"], 3),
        "conf_buy": c["buy"],
        "conf_sell": c["sell"],
        "count_buy": int(c["cnt_buy"]),
        "count_sell": int(c["cnt_sell"]),
        "decision": d,
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
    }

@app.get("/size")
def size(symbol: str, entry: float = Query(...), sl: float = Query(...), account: float = Query(...), risk_pct: float = Query(1.0)):
    risk_amt = account * (risk_pct / 100.0)
    sl_dist = abs(entry - sl)
    units = 0.0 if sl_dist <= 0 else risk_amt / sl_dist
    return {"symbol": symbol, "entry": entry, "sl": sl, "account": account, "risk_pct": risk_pct, "units": round(units, 6)}

@app.post("/scan_now_strict")
def scan_now_strict():
    sent = []
    for sym in SYMBOLS:
        res = analyze_strict(sym)
        if res["decision"] in ("BUY","SELL"):
            txt = f"*{sym}* {res['decision']}\nPrice: {res['price']}\nP(up): {res['p_up']}  P(down): {res['p_down']}\nConf: buy={res['count_buy']} sell={res['count_sell']}\nSL: {res['sl']}  TP1: {res['tp1']}  TP2: {res['tp2']}\n{INTERVAL}  {res['time']}"
            try:
                asyncio.get_event_loop().run_until_complete(send_telegram(txt))
                sent.append(sym)
            except Exception:
                pass
    return {"ok": True, "alerts": sent}
