import websocket
import json
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import os
import logging

# 1. Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# 2. Загрузка API ключей из .env файла
load_dotenv()  # Убедитесь, что файл .env находится в корневой папке проекта
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")


# Проверяем, загружены ли ключи
if not API_KEY or not API_SECRET:
    logger.error("API ключи не найдены. Проверьте файл .env!")
else:
    logger.info("API ключи успешно загружены.")

client = Client(API_KEY, API_SECRET)

# 3. Функция получения исторических данных
def get_historical_data(symbol, interval="1d", start_str="30 days ago UTC"):
    """
    Получение исторических данных с Binance.
    """
    try:
        logger.info(f"Запрос исторических данных для {symbol} с интервалом {interval} за {start_str}")
        candles = client.get_historical_klines(symbol, interval, start_str)
        df = pd.DataFrame(candles, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        logger.error(f"Ошибка при получении исторических данных для {symbol}: {e}")
        return pd.DataFrame()  # Возвращаем пустой DataFrame при ошибке

# 4. Функция подключения к WebSocket Binance
def connect_websocket(symbol, interval="1m"):
    """
    Подключение к WebSocket Binance для получения данных в реальном времени.
    """
    try:
        url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
        ws = websocket.create_connection(url)
        logger.info(f"Подключение к WebSocket установлено для {symbol} с интервалом {interval}")
        return ws
    except Exception as e:
        logger.error(f"Ошибка подключения к WebSocket для {symbol}: {e}")
        return None  # Возвращаем None при сбое

# 5. Функция получения данных в реальном времени через WebSocket
def fetch_real_time_data(ws):
    """
    Получение данных в реальном времени через WebSocket.
    """
    try:
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
    except Exception as e:
        logger.error(f"Ошибка при получении данных в реальном времени: {e}")
        return None  # Возвращаем None при ошибке

# 6. Пример использования
if __name__ == "__main__":
    # Пример работы с историческими данными
    symbol = "BTCUSDT"
    interval = "1d"
    start = "30 days ago UTC"
    historical_data = get_historical_data(symbol, interval, start)
    if not historical_data.empty:
        print(historical_data.head())
    else:
        logger.warning("Исторические данные не получены.")

    # Пример работы с WebSocket
    ws = connect_websocket(symbol, interval="1m")
    if ws:
        for _ in range(5):  # Получение 5 обновлений
            data = fetch_real_time_data(ws)
            if data:
                print(data)
        ws.close()
        logger.info("Соединение с WebSocket закрыто.")
