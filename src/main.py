import os, json, time, urllib.request, urllib.parse
from fastapi import FastAPI, Query

app = FastAPI(title="Sniper Data API")
ALPHAV_KEY = os.getenv("ALPHAV_KEY", "5RVL02E162SRP4G7")

def _get(url, params=None, timeout=15):
    if params: url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

def fetch_binance(symbol: str):
    pair = symbol.upper()
    if not pair.endswith("USDT"): pair += "USDT"
    j = _get("https://api.binance.com/api/v3/klines", {"symbol": pair, "interval": "1m", "limit": 1})
    if not j: return {"source":"binance","values":[]}
    k = j[0]
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(k[0]/1000)))
    return {"source":"binance","values":[{"datetime":t,"close":k[4]}],"last":t}

def fetch_yahoo(symbol: str):
    j = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}",
             {"range":"1d","interval":"1m"})
    r = (j.get("chart",{}) or {}).get("result",[{}])[0]
    ts = r.get("timestamp") or []
    q = (r.get("indicators",{}) or {}).get("quote",[{}])[0]
    if not ts: return {"source":"yahoo","values":[]}
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts[-1]))
    close = (q.get("close") or [None])[-1]
    return {"source":"yahoo","values":[{"datetime":t,"close":close}],"last":t}

def fetch_alphav_fx(symbol: str):
    base, quote = symbol[:3].upper(), symbol[3:].upper()
    j = _get("https://www.alphavantage.co/query",
             {"function":"FX_INTRADAY","from_symbol":base,"to_symbol":quote,
              "interval":"5min","apikey":ALPHAV_KEY})
    ts = j.get("Time Series FX (5min)") or {}
    if not ts: return {"source":"alphav","values":[],"error":j.get("Note") or j.get("Error Message")}
    t, v = next(iter(ts.items()))
    return {"source":"alphav","values":[{"datetime":t,"close":v.get("4. close")}],"last":t}

def fetch_any(symbol: str):
    s = symbol.strip().upper()
    if s in {"BTC","ETH"} or s.endswith("USDT"):
        return fetch_binance(s)
    if len(s) == 6 and s.isalpha():
        return fetch_alphav_fx(s)
    return fetch_yahoo(symbol)

@app.get("/health")
def health():
    return {"ok": True, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}

@app.get("/routes")
def routes():
    return [r.path for r in app.routes]

@app.get("/debug")
def debug(symbol: str = Query(...)):
    r = fetch_any(symbol)
    return {
        "symbol": symbol,
        "source": r.get("source"),
        "rows": len(r.get("values", [])),
        "last": r.get("last"),
        "error": r.get("error")
    }

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    r = fetch_any(symbol)
    vals = r.get("values", [])
    last = vals[0] if vals else {}
    price = float(last.get("close") or 0.0) if last else 0.0
    t = last.get("datetime") if last else ""
    return {
        "symbol": symbol,
        "time": t,
        "price": price,
        "source": r.get("source"),
        "rule_signal": 0,
        "ml_pred": 0,
        "p_up": 0.0,
        "p_down": 0.0,
        "decision": "HOLD",
        "sl": 0.0,
        "tp1": 0.0,
        "tp2": 0.0
    }
