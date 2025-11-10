import os, datetime as dt, requests
from fastapi import FastAPI, Query

app = FastAPI(title="Sniper Data API")
TD_KEY = os.getenv("TWELVEDATA_KEY","").strip()

# symbol mapping
YF_MAP = {
    "XAU/USD": ["XAUUSD=X","GC=F"],
    "BTC/USD": ["BTC-USD"],
    "ETH/USD": ["ETH-USD"],
}

def td_fetch(symbol: str, interval: str="1h", size: int=200):
    if not TD_KEY:
        return {"source":"twelve","values":[], "last":None, "error":"TWELVEDATA_KEY not set"}
    url = "https://api.twelvedata.com/time_series"
    p = {"symbol":symbol, "interval":interval, "outputsize":size, "apikey":TD_KEY}
    try:
        r = requests.get(url, params=p, timeout=15)
        j = r.json()
        vals = j.get("values", []) or []
        last = vals[0] if vals else None   # TD returns newest first
        err  = None if vals else j.get("message","no data")
        return {"source":"twelve","values":vals,"last":last,"error":err}
    except Exception as e:
        return {"source":"twelve","values":[], "last":None, "error":str(e)}

def yf_fetch(yf_symbol: str, interval: str="1h", range_: str="2d"):
    # Yahoo chart JSON, newest in result[0]["timestamp"][-1]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}"
    p = {"interval":interval, "range":range_}
    try:
        r = requests.get(url, params=p, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        j = r.json()
        res = j.get("chart",{}).get("result")
        if not res: 
            err = j.get("chart",{}).get("error",{}).get("description","no data")
            return {"source":"yahoo","values":[], "last":None, "error":err}
        res = res[0]
        ts  = res.get("timestamp",[]) or []
        ind = res.get("indicators",{}).get("quote",[{}])[0]
        closes = ind.get("close",[]) or []
        if not ts or not closes: 
            return {"source":"yahoo","values":[], "last":None, "error":"empty series"}
        # newest
        t = ts[-1]
        c = closes[-1]
        last_iso = dt.datetime.utcfromtimestamp(t).isoformat()+"Z"
        return {"source":"yahoo","values":[{"datetime":last_iso,"close":c}], "last":{"datetime":last_iso,"close":c}, "error":None}
    except Exception as e:
        return {"source":"yahoo","values":[], "last":None, "error":str(e)}

def fetch_any(symbol: str, interval: str="1h"):
    # 1) Try TwelveData first
    td = td_fetch(symbol, interval)
    if td["values"]:
        return td
    # 2) Yahoo fallback for known symbols
    for yf_sym in YF_MAP.get(symbol.upper(), []):
        y = yf_fetch(yf_sym, interval=("15m" if interval=="1h" else interval))
        if y["values"]:
            return y | {"yf_symbol": yf_sym}
    return td  # return TD error if all failed

@app.get("/health")
def health():
    return {"ok":True, "time":dt.datetime.utcnow().isoformat()+"Z"}

@app.get("/routes")
def routes():
    return [r.path for r in app.router.routes]

@app.get("/debug")
def debug(symbol: str = Query(...), interval: str = Query("1h")):
    res = fetch_any(symbol, interval)
    return {
        "symbol":symbol, "interval":interval, "source":res.get("source"),
        "rows":len(res.get("values",[])), "last":res.get("last"),
        "yf_symbol":res.get("yf_symbol"), "error":res.get("error")
    }

@app.get("/analyze")
def analyze(symbol: str = Query(...), interval: str = Query("1h")):
    res = fetch_any(symbol, interval)
    vals = res.get("values",[])
    price = float(vals[0]["close"]) if vals and "close" in vals[0] else 0.0
    t = vals[0].get("datetime","") if vals else ""
    return {
        "symbol":symbol, "interval":interval, "source":res.get("source"),
        "time":t, "price":price,
        "decision":"HOLD"
    }
