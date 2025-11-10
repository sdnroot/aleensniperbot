import os, time, json
from typing import List, Dict, Any
import requests
from fastapi import FastAPI, Query

ALPHAV_KEY = os.getenv("ALPHAV_KEY", "")
UA = {"User-Agent":"Mozilla/5.0"}

app = FastAPI(title="Sniper AI Data API")

def _ok(values: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not values:
        return {"values": [], "last": None}
    # ensure newest first
    values = sorted(values, key=lambda x: x["datetime"], reverse=True)
    return {"values": values, "last": values[0]["datetime"]}

def fetch_alphav_xauusd() -> Dict[str, Any]:
    if not ALPHAV_KEY:
        return {"error":"ALPHAV_KEY missing"}
    try:
        # Intraday FX for XAU/USD
        url = "https://www.alphavantage.co/query"
        params = {"function":"FX_INTRADAY","from_symbol":"XAU","to_symbol":"USD","interval":"5min","apikey":ALPHAV_KEY,"outputsize":"compact"}
        r = requests.get(url, params=params, timeout=15)
        j = r.json()
        ts = j.get("Time Series FX (5min)") or {}
        vals = []
        for t, ohlc in ts.items():
            vals.append({"datetime": t, "close": float(ohlc["4. close"])})
        out = _ok(vals)
        out["source"] = "alphav"
        return out
    except Exception as e:
        return {"error": str(e)}

def fetch_yahoo(symbol: str) -> Dict[str, Any]:
    try:
        # Yahoo chart API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"range":"1d","interval":"1m"}
        j = requests.get(url, params=params, headers=UA, timeout=15).json()
        res = j.get("chart", {}).get("result", [])
        if not res:
            return {"error":"no yahoo result"}
        r0 = res[0]
        ts = r0.get("timestamp") or []
        ind = r0.get("indicators", {}).get("quote", [{}])[0]
        closes = ind.get("close") or []
        vals = []
        for t, c in zip(ts, closes):
            if c is None: 
                continue
            iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
            vals.append({"datetime": iso, "close": float(c)})
        out = _ok(vals)
        out["source"] = "yahoo"
        return out
    except Exception as e:
        return {"error": str(e)}

def fetch_binance(symbol: str) -> Dict[str, Any]:
    try:
        # Map common symbols to Binance
        mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
        }
        sy = mapping.get(symbol.upper(), None)
        if not sy:
            return {"error":"binance symbol not mapped"}
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": sy, "interval":"1m", "limit": 50}
        arr = requests.get(url, params=params, timeout=15).json()
        vals = []
        for k in arr:
            # kline: [open_time, open, high, low, close, ...]
            ts = int(k[0]) // 1000
            iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
            vals.append({"datetime": iso, "close": float(k[4])})
        out = _ok(vals)
        out["source"] = "binance"
        return out
    except Exception as e:
        return {"error": str(e)}

def fetch_any(symbol: str) -> Dict[str, Any]:
    s = symbol.upper()
    # 1) XAUUSD → AlphaVantage
    if s in ("XAUUSD", "XAU/USD", "XAUUSD=X", "XAU-USD"):
        av = fetch_alphav_xauusd()
        if av.get("values"):
            return av
        # fallback to Yahoo: GC=F futures
        ya = fetch_yahoo("GC=F")
        if ya.get("values"):
            return ya
        return {"error": av.get("error") or ya.get("error") or "no data"}

    # 2) Futures or Yahoo-tickers directly
    if s in ("GC=F","XAUUSD=X","BTC-USD","ETH-USD"):
        ya = fetch_yahoo(s)
        if ya.get("values"):
            return ya
        return {"error": ya.get("error") or "no data"}

    # 3) Crypto short symbol → Binance
    if s in ("BTC","ETH"):
        bz = fetch_binance(s)
        if bz.get("values"):
            return bz
        # fallback to Yahoo crypto
        ya = fetch_yahoo(f"{s}-USD")
        if ya.get("values"):
            return ya
        return {"error": bz.get("error") or ya.get("error") or "no data"}

    # Generic Yahoo final try
    ya = fetch_yahoo(s)
    if ya.get("values"):
        return ya
    return {"error": ya.get("error") or "no data"}

@app.get("/health")
def health():
    return {"ok": True, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

@app.get("/routes")
def routes():
    return ["/openapi.json","/docs","/health","/routes","/debug","/analyze"]

@app.get("/debug")
def debug(symbol: str = Query(...)):
    res = fetch_any(symbol)
    return {
        "symbol": symbol,
        "source": res.get("source"),
        "rows": len(res.get("values", [])),
        "last": res.get("last"),
        "error": res.get("error")
    }

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    res = fetch_any(symbol)
    vals = res.get("values", [])
    price = float(vals[0]["close"]) if vals else 0.0
    t = vals[0]["datetime"] if vals else ""
    return {
        "symbol": symbol,
        "time": t,
        "price": price,
        "source": res.get("source"),
        "rule_signal": 0, "ml_pred": 0, "p_up": 0.0, "p_down": 0.0,
        "decision": "HOLD", "sl": 0.0, "tp1": 0.0, "tp2": 0.0
    }
