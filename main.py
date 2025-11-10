import os, requests, pandas as pd
from fastapi import FastAPI, Query

app = FastAPI()

TD_KEY = os.getenv("TWELVEDATA_KEY", "1568d10968484808a32195ae759c0a17")
MAP = {
    "GC=F": "XAU/USD",
    "XAUUSD=X": "XAU/USD",
    "XAUUSD": "XAU/USD",
    "XAU/USD": "XAU/USD",
    "BTC-USD": "BTC/USD",
    "BTC/USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "ETH/USD": "ETH/USD",
}

def fetch_prices_td(symbol: str, interval: str = "15min") -> pd.DataFrame:
    sym = MAP.get(symbol, symbol)
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": sym, "interval": interval, "outputsize": 200, "apikey": TD_KEY},
            timeout=20,
        )
        j = r.json()
        v = j.get("values") or []
        if not v:
            return pd.DataFrame()
        df = pd.DataFrame(v)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.rename(
            columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}
        )
        return df
    except Exception:
        return pd.DataFrame()

@app.get("/health")
def health():
    return {"ok": True, "time": pd.Timestamp.utcnow().isoformat()}

@app.get("/debug")
def debug(symbol: str = Query(...)):
    df = fetch_prices_td(symbol)
    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "cols": (list(df.columns) if len(df) > 0 else []),
        "last": (df.index[-1].isoformat() if len(df) > 0 else None),
    }

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    df = fetch_prices_td(symbol)
    price = float(df["Close"].iloc[-1]) if len(df) > 0 and "Close" in df.columns else 0.0
    t = df.index[-1].isoformat() if len(df) > 0 else ""
    # placeholders for now
    rule_signal = 0
    ml_pred = 0
    p_up = 0.0
    p_down = 0.0
    decision = "HOLD"
    sl = tp1 = tp2 = 0.0
    return {
        "symbol": symbol, "time": t, "price": price,
        "rule_signal": rule_signal, "ml_pred": ml_pred,
        "p_up": p_up, "p_down": p_down, "decision": decision,
        "sl": sl, "tp1": tp1, "tp2": tp2
    }

import os, requests, pandas as pd
_TD_KEY = os.getenv("TWELVEDATA_KEY","1568d10968484808a32195ae759c0a17")
_SYM = {"GC=F":"XAU/USD","XAUUSD=X":"XAU/USD","XAUUSD":"XAU/USD","XAU/USD":"XAU/USD",
        "BTC-USD":"BTC/USD","BTC/USD":"BTC/USD","ETH-USD":"ETH/USD","ETH/USD":"ETH/USD"}
def _td_fetch(symbol:str, interval:str="1h"):
    sym = _SYM.get(symbol, symbol)
    try:
        r = requests.get("https://api.twelvedata.com/time_series",
                         params={"symbol":sym,"interval":interval,"outputsize":200,"apikey":_TD_KEY},
                         timeout=20)
        j = r.json(); v = j.get("values") or []
        if not v: return pd.DataFrame()
        df = pd.DataFrame(v)
        df["datetime"]=pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        for c in ["open","high","low","close","volume"]:
            df[c]=pd.to_numeric(df[c], errors="coerce")
        return df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    except Exception:
        return pd.DataFrame()
