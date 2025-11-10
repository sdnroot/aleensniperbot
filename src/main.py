import os, datetime as dt, requests
from fastapi import FastAPI, Query

app = FastAPI(title="Sniper AI Minimal", version="1.0")
TD_KEY = os.getenv("TWELVEDATA_KEY","")

ALIASES = {
    "GC=F":"XAU/USD","XAUUSD=X":"XAU/USD","XAUUSD":"XAU/USD","XAU/USD":"XAU/USD",
    "BTC-USD":"BTC/USD","BTC/USD":"BTC/USD","ETH-USD":"ETH/USD","ETH/USD":"ETH/USD",
}
def _norm(s:str)->str:
    return ALIASES.get((s or "").strip().upper().replace("%2F","/").replace("\\","/"), (s or "").strip())

def td_fetch(symbol:str, interval:str="1h", size:int=200):
    if not TD_KEY:
        return {"error":"no TWELVEDATA_KEY set"}
    url="https://api.twelvedata.com/time_series"
    params={"symbol":_norm(symbol),"interval":interval,"outputsize":size,"apikey":TD_KEY}
    r = requests.get(url, params=params, timeout=20)
    j = r.json()
    vals = j.get("values") or []
    if not vals:
        return {"values":[], "meta": j.get("meta",{})}
    last = vals[0] if isinstance(vals,list) and len(vals)>0 else vals[-1]
    return {"values": vals, "last": last, "meta": j.get("meta",{})}

@app.get("/health")
def health():
    return {"ok": True, "time": dt.datetime.utcnow().isoformat() + "Z"}

@app.get("/routes")
def routes():
    return [r.path for r in app.router.routes]

@app.get("/debug")
def debug(symbol: str = Query(...), interval: str = Query("1h")):
    res = td_fetch(symbol, interval=interval, size=200)
    return {"symbol": _norm(symbol), "interval": interval, "rows": len(res.get("values",[])), "last": res.get("last"), "meta": res.get("meta")}

@app.get("/analyze")
def analyze(symbol: str = Query(...), interval: str = Query("1h")):
    res = td_fetch(symbol, interval=interval, size=200)
    vals = res.get("values", [])
    price = float(vals[0]["close"]) if vals and "close" in vals[0] else 0.0
    t = vals[0].get("datetime") if vals and "datetime" in vals[0] else ""
    return {"symbol": _norm(symbol), "interval": interval, "time": t, "price": price,
            "rule_signal": 0, "ml_pred": 0, "p_up": 0.0, "p_down": 0.0,
            "decision": "HOLD", "sl": 0.0, "tp1": 0.0, "tp2": 0.0}
