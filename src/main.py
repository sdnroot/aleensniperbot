import os, datetime as dt, requests
from fastapi import FastAPI, Query

app = FastAPI(title="Sniper AI Minimal", version="1.0")
TD_KEY = os.getenv("TWELVEDATA_KEY","")

def td_fetch(symbol:str, interval:str="1h", size:int=200):
    url="https://api.twelvedata.com/time_series"
    params={"symbol":symbol,"interval":interval,"outputsize":size,"apikey":TD_KEY}
    try:
        r=requests.get(url,params=params,timeout=15)
        j=r.json()
        vals=j.get("values",[])
        if not vals:
            return {"values":[],"meta":j.get("meta",{}),"error":j.get("message")}
        last=vals[0]
        return {"values":vals,"last":last}
    except Exception as e:
        return {"values":[],"error":str(e)}

@app.get("/health")
def health():
    return {"ok":True,"time":dt.datetime.utcnow().isoformat()+"Z"}

@app.get("/routes")
def routes():
    return [r.path for r in app.router.routes]

@app.get("/debug")
def debug(symbol:str=Query(...),interval:str=Query("1h")):
    res=td_fetch(symbol,interval)
    return {"symbol":symbol,"interval":interval,"rows":len(res.get("values",[])),"last":res.get("last"),"error":res.get("error")}

@app.get("/analyze")
def analyze(symbol:str=Query(...),interval:str=Query("1h")):
    res=td_fetch(symbol,interval)
    vals=res.get("values",[])
    price=float(vals[0]["close"]) if vals and "close" in vals[0] else 0.0
    t=vals[0].get("datetime") if vals else ""
    return {"symbol":symbol,"interval":interval,"time":t,"price":price,"decision":"HOLD"}
