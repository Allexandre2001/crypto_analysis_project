import websocket
import json
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import os
import logging

from data.calculations import clean_data

# 1. Налаштування логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# 2. Завантаження API ключів із .env файлу
load_dotenv()  # Переконайтеся, що файл .env знаходиться в кореневій папці проекту
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")


# Перевіряємо, чи завантажені ключі
if not API_KEY or not API_SECRET:
    logger.error("API ключи не найдены. Проверьте файл .env!")
else:
    logger.info("API ключи успешно загружены.")

client = Client(API_KEY, API_SECRET)

# 3. Функція отримання історичних даних
def get_historical_data(symbol, interval="1d", start_str="30 days ago UTC"):
    """
       Завантаження історичних даних для криптовалютної пари.

       Аргументи:
       - pair: Назва криптовалютної пари (наприклад, BTCUSDT).
       - interval: Часовий інтервал (наприклад, 1d, 1h).
       - start_str: Час початку у форматі рядка (наприклад, "30 days ago UTC").

       Повертає:
       - DataFrame з історичними даними.
       """
    try:
        logger.info(f"Запит історичних даних для {symbol} з інтервалом {interval} за {start_str}")
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
        logger.error(f"Помилка завантаження даних для: {symbol}: {e}")
        return pd.DataFrame()  # Повертаємо порожній DataFrame при помилці

# 4. Функція підключення до WebSocket Binance
def connect_websocket(symbol, interval="1m"):
    """
 Підключення до WebSocket Binance для отримання даних у реальному часі.
    """
    try:
        url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
        ws = websocket.create_connection(url)
        logger.info(f"Підключення до WebSocket встановлено для {symbol} з інтервалом {interval}")
        return ws
    except Exception as e:
        logger.error(f"Помилка підключення до WebSocket для {symbol}: {e}")
        return None  # Повертаємо None під час збою

# 5. Функція отримання даних у реальному часі через WebSocket
def fetch_real_time_data(ws):
    """
    Отримання даних у реальному часі через WebSocket.
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
        logger.error(f"Помилка при отриманні даних у реальному часі: {e}")
        return None  # Повертаємо None під час збою

def get_historical_data_cleaned(symbol, interval="1d", start_str="30 days ago UTC"):
    """
    Отримання історичних даних із очищенням.
    """
    raw_data = get_historical_data(symbol, interval, start_str)
    if raw_data.empty:
        return pd.DataFrame() # Повертаємо порожній DataFrame при помилці
    return clean_data(raw_data)