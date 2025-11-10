import os, requests, datetime as dt
from fastapi import FastAPI, Query

app = FastAPI(title="Sniper TD API")

TD_KEY = os.getenv("TWELVEDATA_KEY","")

def td_fetch(symbol: str, interval: str = "1h", size: int = 200):
    url = "https://api.twelvedata.com/time_series"
    p = {"symbol": symbol, "interval": interval, "outputsize": size, "apikey": TD_KEY}
    try:
        r = requests.get(url, params=p, timeout=15)
        j = r.json()
        vals = j.get("values", [])
        err = j.get("message") if not vals else None
        last = vals[0] if vals else None  # TD returns newest first
        return {"values": vals, "last": last, "error": err}
    except Exception as e:
        return {"values": [], "last": None, "error": str(e)}

@app.get("/health")
def health():
    return {"ok": True, "time": dt.datetime.utcnow().isoformat()+"Z"}

@app.get("/routes")
def list_routes():
    return [r.path for r in app.router.routes]

@app.get("/debug")
def debug(symbol: str = Query(...), interval: str = Query("1h")):
    res = td_fetch(symbol, interval)
    return {
        "symbol": symbol,
        "interval": interval,
        "rows": len(res["values"]),
        "last": res["last"],
        "error": res["error"],
    }

@app.get("/analyze")
def analyze(symbol: str = Query(...), interval: str = Query("1h")):
    res = td_fetch(symbol, interval)
    vals = res["values"]
    price = float(vals[0]["close"]) if vals and "close" in vals[0] else 0.0
    t = vals[0].get("datetime") if vals else ""
    return {
        "symbol": symbol,
        "interval": interval,
        "time": t,
        "price": price,
        "decision": "HOLD"
    }
