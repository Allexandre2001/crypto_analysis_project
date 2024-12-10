import websocket
import json
import pandas as pd
from binance.client import Client

def get_historical_data(symbol, interval="1d", start_str="30 days ago UTC"):
    """
    Получение исторических данных с Binance.
    """
    API_KEY = ""  # Ваш API ключ
    API_SECRET = ""  # Ваш API секрет
    client = Client(API_KEY, API_SECRET)

    candles = client.get_historical_klines(symbol, interval, start_str)
    df = pd.DataFrame(candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def connect_websocket(symbol):
    """
    Подключение к WebSocket Binance для получения данных в реальном времени.
    """
    url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_1m"
    ws = websocket.create_connection(url)
    return ws

def fetch_real_time_data(ws):
    """
    Получение данных в реальном времени через WebSocket.
    """
    message = ws.recv()
    data = json.loads(message)
    kline = data["k"]
    return {
        "open_time": pd.to_datetime(kline["t"], unit="ms"),
        "open": float(kline["o"]),
        "high": float(kline["h"]),
        "low": float(kline["l"]),
        "close": float(kline["c"]),
        "volume": float(kline["v"]),
    }
