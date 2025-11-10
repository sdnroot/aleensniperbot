import os, json, time, urllib.request, urllib.parse
from fastapi import FastAPI, Query

app = FastAPI(title="Sniper Data API")
ALPHAV_KEY = os.getenv("ALPHAV_KEY", "5RVL02E162SRP4G7")

def _get(url, params=None):
    if params: url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode())

def fetch_binance(symbol):
    pair = symbol.upper() + "USDT" if not symbol.upper().endswith("USDT") else symbol.upper()
    j = _get("https://api.binance.com/api/v3/klines", {"symbol":pair,"interval":"1m","limit":1})
    if not j: return {"source":"binance","values":[]}
    k = j[0]
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(k[0]/1000)))
    return {"source":"binance","values":[{"datetime":t,"close":k[4]}],"last":t}

def fetch_yahoo(symbol):
    j = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",{"range":"1d","interval":"1m"})
    r = (j.get("chart",{}) or {}).get("result",[{}])[0]
    ts, quotes = r.get("timestamp") or [], (r.get("indicators",{}) or {}).get("quote",[{}])[0]
    if not ts: return {"source":"yahoo","values":[]}
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts[-1]))
    return {"source":"yahoo","values":[{"datetime":t,"close":quotes.get('close',[-1])[-1]}],"last":t}

def fetch_alphav(symbol):
    j = _get("https://www.alphavantage.co/query",
             {"function":"FX_INTRADAY","from_symbol":symbol[:3],"to_symbol":symbol[3:],
              "interval":"5min","apikey":ALPHAV_KEY})
    ts = j.get("Time Series FX (5min)") or {}
    if not ts: return {"source":"alphav","values":[]}
    t, v = next(iter(ts.items()))
    return {"source":"alphav","values":[{"datetime":t,"close":v.get("4. close")}],"last":t}

@app.get("/health")
def health(): return {"ok":True,"time":time.strftime("%Y-%m-%dT%H:%M:%S")}

@app.get("/routes")
def routes(): return [r.path for r in app.routes]

@app.get("/debug")
def debug(symbol:str=Query(...)):
    if symbol.upper() in {"BTC","ETH"}: r=fetch_binance(symbol)
    elif len(symbol)==6: r=fetch_alphav(symbol)
    else: r=fetch_yahoo(symbol)
    return {"symbol":symbol,"source":r.get("source"),"rows":len(r.get("values",[])),"last":r.get("last")}

@app.get("/analyze")
def analyze(symbol:str=Query(...)):
    r = debug(symbol)
    price = r.get("values",[{}])[0].get("close",0.0) if r.get("values") else 0.0
    return {"symbol":symbol,"price":price,"source":r.get("source")}
