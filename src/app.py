import os, httpx
from fastapi import FastAPI, Query

ALPHA = os.getenv("ALPHAV_KEY","")
app = FastAPI(title="Sniper AI Bot")

def alpha_fx(symbol: str, interval: str="15min", outputsize: str="compact"):
    if not ALPHA: 
        return {"source":"alpha","values":[], "error":"ALPHAV_KEY missing"}
    s = symbol.replace("-","").replace("/","").upper()
    if len(s) < 6: return {"source":"alpha","values":[], "error":"bad symbol"}
    from_symbol, to_symbol = s[:3], s[3:]
    params = dict(function="FX_INTRADAY", from_symbol=from_symbol, to_symbol=to_symbol,
                  interval=interval, outputsize=outputsize, apikey=ALPHA)
    try:
        with httpx.Client(timeout=15) as c:
            r = c.get("https://www.alphavantage.co/query", params=params)
        j = r.json()
        key = next((k for k in j.keys() if "Time Series" in k), None)
        if not key: 
            return {"source":"alpha","values":[], "error":j.get("Note") or j.get("Error Message") or "no series"}
        series = j[key]
        vals = []
        for ts, row in series.items():
            vals.append({
                "datetime": ts,
                "open": float(row.get("1. open", 0.0)),
                "high": float(row.get("2. high", 0.0)),
                "low":  float(row.get("3. low", 0.0)),
                "close":float(row.get("4. close", 0.0)),
                "volume": float(row.get("5. volume", 0.0)) if "5. volume" in row else 0.0
            })
        vals.sort(key=lambda x: x["datetime"], reverse=True)
        return {"source":"alpha","values":vals, "last": (vals[0]["datetime"] if vals else None)}
    except Exception as e:
        return {"source":"alpha","values":[], "error":str(e)}

def alpha_crypto(symbol: str, interval: str="5min"):
    if not ALPHA:
        return {"source":"alpha","values":[], "error":"ALPHAV_KEY missing"}
    s = symbol.replace("-","").replace("/","").upper()
    if not s.endswith("USD"): return {"source":"alpha","values":[], "error":"use BTCUSD/ETHUSD"}
    params = dict(function="CRYPTO_INTRADAY", symbol=s[:-3], market=s[-3:], interval=interval, apikey=ALPHA)
    try:
        with httpx.Client(timeout=15) as c:
            r = c.get("https://www.alphavantage.co/query", params=params)
        j = r.json()
        key = next((k for k in j.keys() if "Time Series" in k), None)
        if not key:
            return {"source":"alpha","values":[], "error":j.get("Note") or j.get("Error Message") or "no series"}
        series = j[key]
        vals = []
        for ts, row in series.items():
            vals.append({
                "datetime": ts,
                "open": float(row.get("1. open", 0.0)),
                "high": float(row.get("2. high", 0.0)),
                "low":  float(row.get("3. low", 0.0)),
                "close":float(row.get("4. close", 0.0)),
                "volume": float(row.get("5. volume", 0.0)) if "5. volume" in row else 0.0
            })
        vals.sort(key=lambda x: x["datetime"], reverse=True)
        return {"source":"alpha","values":vals, "last": (vals[0]["datetime"] if vals else None)}
    except Exception as e:
        return {"source":"alpha","values":[], "error":str(e)}

def fetch_any(symbol: str):
    s = symbol.replace("-","").replace("/","").upper()
    if s in ("XAUUSD","XAGUSD") or (len(s)==6 and s.endswith("USD") and not s.startswith(("BTC","ETH","SOL","XRP","ADA","DOGE","BNB","TON"))):
        return alpha_fx(s)
    if s.endswith("USD"):
        return alpha_crypto(s)
    return {"source":"none","values":[], "error":"unsupported symbol"}

@app.get("/health")
def health(): return {"ok": True}

@app.get("/routes")
def routes(): return ["/health","/routes","/debug","/analyze"]

@app.get("/debug")
def debug(symbol: str = Query(...)):
    r = fetch_any(symbol)
    return {"symbol":symbol, "source":r.get("source"), "rows":len(r.get("values",[])),
            "last":r.get("last"), "error":r.get("error")}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    r = fetch_any(symbol)
    vals = r.get("values", [])
    last = vals[0] if vals else {}
    price = float(last.get("close", 0.0) or 0.0) if last else 0.0
    t = last.get("datetime") if last else ""
    return {"symbol":symbol, "time":t, "price":price, "source":r.get("source"),
            "rule_signal":0, "ml_pred":0, "p_up":0.0, "p_down":0.0,
            "decision":"HOLD", "sl":0.0, "tp1":0.0, "tp2":0.0}
