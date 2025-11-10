import os, datetime as dt, requests, pandas as pd
from fastapi import FastAPI, Query

app = FastAPI(title="Sniper AI Bot", version="1.0")

TD_KEY = os.getenv("TWELVEDATA_KEY","")

ALIASES = {
    "GC=F":"XAU/USD","XAUUSD=X":"XAU/USD","XAUUSD":"XAU/USD","XAU/USD":"XAU/USD",
    "BTC-USD":"BTC/USD","BTC/USD":"BTC/USD","ETH-USD":"ETH/USD","ETH/USD":"ETH/USD",
}
def _norm(s:str)->str:
    s=(s or "").strip().upper().replace("%2F","/").replace("\\","/")
    return ALIASES.get(s,s)

def td_fetch(symbol:str, interval:str="1h", size:int=200)->pd.DataFrame:
    if not TD_KEY: return pd.DataFrame()
    url="https://api.twelvedata.com/time_series"
    params={"symbol":_norm(symbol),"interval":interval,"outputsize":size,"apikey":TD_KEY}
    try:
        r=requests.get(url,params=params,timeout=20); j=r.json(); v=j.get("values",[])
        if not v: return pd.DataFrame()
        df=pd.DataFrame(v)
        if "datetime" in df.columns:
            df["datetime"]=pd.to_datetime(df["datetime"])
            df=df.sort_values("datetime").set_index("datetime")
        for c in ("open","high","low","close","volume"):
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"},inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@app.get("/health")
def health():
    return {"ok":True,"time":dt.datetime.utcnow().isoformat()+"Z"}

@app.get("/debug")
def debug(symbol: str = Query(...), interval: str = Query("1h")):
    df=td_fetch(symbol,interval=interval,size=200)
    last=df.index[-1].isoformat() if len(df)>0 else None
    cols=list(df.columns) if len(df)>0 else []
    return {"symbol":_norm(symbol),"interval":interval,"rows":int(len(df)),"cols":cols,"last":last}

@app.get("/analyze")
def analyze(symbol: str = Query(...), interval: str = Query("1h")):
    df=td_fetch(symbol,interval=interval,size=200)
    price=float(df["Close"].iloc[-1]) if len(df)>0 and "Close" in df.columns else 0.0
    t=df.index[-1].isoformat() if len(df)>0 else ""
    return {"symbol":_norm(symbol),"interval":interval,"time":t,"price":price,
            "rule_signal":0,"ml_pred":0,"p_up":0.0,"p_down":0.0,
            "decision":"HOLD","sl":0.0,"tp1":0.0,"tp2":0.0}
