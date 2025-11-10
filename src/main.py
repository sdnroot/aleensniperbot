import os, datetime as dt, requests
from fastapi import FastAPI, Query

app = FastAPI(title="Sniper Data API v2")

ALPHAV_KEY = os.getenv("ALPHAV_KEY","").strip()

def av_fx_intraday(from_sym:str,to_sym:str="USD",interval:str="5min"):
    if not ALPHAV_KEY:
        return {"source":"alphav","values":[],"last":None,"error":"ALPHAV_KEY not set"}
    url="https://www.alphavantage.co/query"
    p={"function":"FX_INTRADAY","from_symbol":from_sym,"to_symbol":to_sym,
       "interval":interval,"outputsize":"compact","apikey":ALPHAV_KEY}
    try:
        r=requests.get(url,params=p,timeout=15)
        j=r.json()
        ts_key=next((k for k in j if k.startswith("Time Series FX")),None)
        if not ts_key: return {"source":"alphav","values":[],"last":None,"error":j.get("Note") or j.get("Error Message")}
        ts=j[ts_key]; t_iso=sorted(ts.keys())[-1]; c=float(ts[t_iso]["4. close"])
        return {"source":"alphav","values":[{"datetime":t_iso,"close":c}],
                "last":{"datetime":t_iso,"close":c},"error":None}
    except Exception as e: return {"source":"alphav","values":[],"last":None,"error":str(e)}

def yahoo_chart(symbol:str,interval:str="15m",range_="2d"):
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        r=requests.get(url,params={"interval":interval,"range":range_},timeout=15,
                       headers={"User-Agent":"Mozilla/5.0"})
        j=r.json()
        res=(j.get("chart",{}) or {}).get("result")
        if not res: return {"source":"yahoo","values":[],"last":None,"error":"no data"}
        res=res[0]; ts=res.get("timestamp",[]); q=res.get("indicators",{}).get("quote",[{}])[0]
        if not ts or not q.get("close"): return {"source":"yahoo","values":[],"last":None,"error":"empty"}
        t_iso=dt.datetime.utcfromtimestamp(ts[-1]).isoformat()+"Z"; c=float(q["close"][-1])
        return {"source":"yahoo","values":[{"datetime":t_iso,"close":c}],
                "last":{"datetime":t_iso,"close":c},"error":None}
    except Exception as e: return {"source":"yahoo","values":[],"last":None,"error":str(e)}

def binance(symbol="BTCUSDT",interval="5m"):
    url="https://api.binance.com/api/v3/klines"
    try:
        r=requests.get(url,params={"symbol":symbol,"interval":interval,"limit":1},timeout=15)
        k=r.json()[0]; t_iso=dt.datetime.utcfromtimestamp(k[0]/1000).isoformat()+"Z"
        return {"source":"binance","values":[{"datetime":t_iso,"close":float(k[4])}],
                "last":{"datetime":t_iso,"close":float(k[4])},"error":None}
    except Exception as e: return {"source":"binance","values":[],"last":None,"error":str(e)}

def fetch_any(symbol:str):
    s=symbol.upper()
    if "XAU" in s: 
        av=av_fx_intraday("XAU","USD")
        if av["values"]: return av
        return yahoo_chart("GC=F")
    if "BTC" in s: return binance("BTCUSDT")
    if "ETH" in s: return binance("ETHUSDT")
    return yahoo_chart(symbol)

@app.get("/health")
def health(): return {"ok":True,"time":dt.datetime.utcnow().isoformat()+"Z"}

@app.get("/routes")
def routes(): return [r.path for r in app.router.routes]

@app.get("/debug")
def debug(symbol:str=Query(...)):
    res=fetch_any(symbol)
    return {"symbol":symbol,"source":res.get("source"),"rows":len(res.get("values",[])),
            "last":res.get("last"),"error":res.get("error")}

@app.get("/analyze")
def analyze(symbol:str=Query(...)):
    res=fetch_any(symbol)
    vals=res.get("values",[])
    price=float(vals[0]["close"]) if vals else 0.0
    t=vals[0]["datetime"] if vals else ""
    return {"symbol":symbol,"time":t,"price":price,"source":res.get("source")}
