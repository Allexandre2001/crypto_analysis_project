import numpy as np
import pandas as pd

def calculate_returns(df):
    """
    Расчёт дневной и кумулятивной доходности.
    """
    df["daily_return"] = df["close"].pct_change()
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df

def calculate_volatility(df):
    """
    Расчёт волатильности (стандартного отклонения дневной доходности).
    """
    return df["daily_return"].std()

def calculate_correlation(data_dict):
    """
    Расчёт корреляции между активами.
    """
    close_prices = {pair: data["close"] for pair, data in data_dict.items()}
    close_df = pd.DataFrame(close_prices)
    return close_df.corr()

def calculate_moving_averages(df, window=14):
    """
    Расчёт скользящих средних (SMA, EMA).
    """
    df["SMA"] = df["close"].rolling(window=window).mean()
    df["EMA"] = df["close"].ewm(span=window, adjust=False).mean()
    return df

def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    """
    Расчёт коэффициента Sharpe для оценки соотношения доходности и риска.
    """
    mean_return = df["daily_return"].mean()
    risk = df["daily_return"].std()
    return (mean_return - risk_free_rate) / risk if risk != 0 else np.nan

def calculate_rsi(df, window=14):
    """
    Расчёт RSI (Relative Strength Index).
    """
    delta = df["close"].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Расчёт MACD и сигнальной линии.
    """
    df["MACD"] = df["close"].ewm(span=short_window, adjust=False).mean() - \
                 df["close"].ewm(span=long_window, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    return df

def generate_advanced_recommendations(data_dict, rsi_thresholds=(30, 70), volume_threshold=1.5):
    """
    Расширенные рекомендации на основе MACD, RSI, объёмов и трендов.
    """
    recommendations = []
    for pair, data in data_dict.items():
        # Рассчитываем индикаторы
        data = calculate_rsi(data)
        data = calculate_macd(data)

        # Последние значения индикаторов
        last_rsi = data["RSI"].iloc[-1]
        last_macd = data["MACD"].iloc[-1]
        last_signal = data["Signal"].iloc[-1]
        last_volume = data["volume"].iloc[-1]
        average_volume = data["volume"].mean()
        last_close = data["close"].iloc[-1]

        # Тренды и сигналы
        trend = "восходящий" if data["close"].iloc[-1] > data["close"].iloc[-2] else "нисходящий"
        volume_status = "увеличен" if last_volume > average_volume * volume_threshold else "нормальный"

        # Рекомендации
        if last_rsi < rsi_thresholds[0] and last_macd > last_signal:
            recommendations.append(
                f"{pair}: Перепродан (RSI: {last_rsi:.2f} < {rsi_thresholds[0]}), сигнал на покупку (MACD > Signal). "
                f"Тренд: {trend}, объём: {volume_status}. Текущая цена: ${last_close:.2f}."
            )
        elif last_rsi > rsi_thresholds[1] and last_macd < last_signal:
            recommendations.append(
                f"{pair}: Перекуплен (RSI: {last_rsi:.2f} > {rsi_thresholds[1]}), сигнал на продажу (MACD < Signal). "
                f"Тренд: {trend}, объём: {volume_status}. Текущая цена: ${last_close:.2f}."
            )
        elif last_macd > last_signal:
            recommendations.append(
                f"{pair}: Сигнал на покупку (MACD > Signal). RSI в норме ({last_rsi:.2f}). "
                f"Тренд: {trend}, объём: {volume_status}. Текущая цена: ${last_close:.2f}."
            )
        elif last_macd < last_signal:
            recommendations.append(
                f"{pair}: Сигнал на продажу (MACD < Signal). RSI в норме ({last_rsi:.2f}). "
                f"Тренд: {trend}, объём: {volume_status}. Текущая цена: ${last_close:.2f}."
            )
        elif last_rsi < rsi_thresholds[0]:
            recommendations.append(
                f"{pair}: Актив перепродан (RSI: {last_rsi:.2f} < {rsi_thresholds[0]}). "
                f"Рекомендуется присмотреться к возможной покупке. Тренд: {trend}, объём: {volume_status}. Текущая цена: ${last_close:.2f}."
            )
        elif last_rsi > rsi_thresholds[1]:
            recommendations.append(
                f"{pair}: Актив перекуплен (RSI: {last_rsi:.2f} > {rsi_thresholds[1]}). "
                f"Рекомендуется зафиксировать прибыль или снизить риски. Тренд: {trend}, объём: {volume_status}. Текущая цена: ${last_close:.2f}."
            )
        else:
            recommendations.append(
                f"{pair}: Актив находится в стабильной зоне (RSI: {last_rsi:.2f}). "
                f"Нет ярко выраженных сигналов для действий. Тренд: {trend}, объём: {volume_status}. Текущая цена: ${last_close:.2f}."
            )
    return recommendations
