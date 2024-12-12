import numpy as np
import pandas as pd
import plotly.graph_objects as go


def calculate_returns(df):
    """
    Расчёт дневной и кумулятивной доходности.

    Параметры:
    df (DataFrame): Данные актива с колонкой "close".

    Возвращает:
    DataFrame: Обновлённый DataFrame с колонками "daily_return" и "cumulative_return".
    """
    if df.empty or "close" not in df:
        raise ValueError("Данные отсутствуют или не содержат колонку 'close'.")
    df["daily_return"] = df["close"].pct_change()
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df

def calculate_volatility(df):
    """
    Рассчитывает волатильность на основе стандартного отклонения дневной доходности.

    Параметры:
    df (DataFrame): Данные актива с колонкой "daily_return".

    Возвращает:
    float: Значение волатильности.
    """
    if df.empty or "daily_return" not in df:
        raise ValueError("Данные отсутствуют или не содержат колонку 'daily_return'.")
    return df["daily_return"].std()

def calculate_correlation(data_dict):
    """
    Рассчитывает корреляцию между активами.

    Параметры:
    data_dict (dict): Словарь с парами активов и их данными.

    Возвращает:
    DataFrame: Матрица корреляции.
    """
    close_prices = {pair: data["close"] for pair, data in data_dict.items()}
    close_df = pd.DataFrame(close_prices)
    return close_df.corr()

def calculate_moving_averages(df, window=14):
    """
    Рассчитывает скользящие средние (SMA, EMA).

    Параметры:
    df (DataFrame): Данные актива с колонкой "close".
    window (int): Период для расчёта SMA и EMA.

    Возвращает:
    DataFrame: Обновлённый DataFrame с колонками "SMA" и "EMA".
    """
    if df.empty or "close" not in df:
        raise ValueError("Данные отсутствуют или не содержат колонку 'close'.")
    df["SMA"] = df["close"].rolling(window=window).mean()
    df["EMA"] = df["close"].ewm(span=window, adjust=False).mean()
    return df

def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    """
    Рассчитывает коэффициент Sharpe для оценки соотношения доходности и риска.

    Параметры:
    df (DataFrame): Данные актива с колонкой "daily_return".
    risk_free_rate (float): Безрисковая ставка.

    Возвращает:
    float: Значение коэффициента Sharpe.
    """
    if df.empty or "daily_return" not in df:
        raise ValueError("Данные отсутствуют или не содержат колонку 'daily_return'.")
    mean_return = df["daily_return"].mean()
    risk = df["daily_return"].std()
    return (mean_return - risk_free_rate) / risk if risk != 0 else np.nan

def calculate_rsi(df, window=14):
    """
    Рассчитывает RSI (Relative Strength Index).

    Параметры:
    df (DataFrame): Данные актива с колонкой "close".
    window (int): Период для расчёта RSI.

    Возвращает:
    DataFrame: Обновлённый DataFrame с колонкой "RSI".
    """
    if df.empty or "close" not in df:
        raise ValueError("Данные отсутствуют или не содержат колонку 'close'.")
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
    Рассчитывает MACD и сигнальную линию.

    Параметры:
    df (DataFrame): Данные актива с колонкой "close".
    short_window (int): Период для быстрого EMA.
    long_window (int): Период для медленного EMA.
    signal_window (int): Период для сигнальной линии.

    Возвращает:
    DataFrame: Обновлённый DataFrame с колонками "MACD" и "Signal".
    """
    if df.empty or "close" not in df:
        raise ValueError("Данные отсутствуют или не содержат колонку 'close'.")
    df["MACD"] = df["close"].ewm(span=short_window, adjust=False).mean() - \
                 df["close"].ewm(span=long_window, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    return df

def generate_advanced_recommendations(data_dict, rsi_thresholds=(30, 70), volume_threshold=1.5):
    """
    Генерирует расширенные рекомендации на основе MACD, RSI, объёмов и трендов.

    Параметры:
    data_dict (dict): Словарь с парами активов и их данными.
    rsi_thresholds (tuple): Пороговые значения RSI для перепроданности и перекупленности.
    volume_threshold (float): Коэффициент для анализа объёма.

    Возвращает:
    list: Список рекомендаций.
    """
    recommendations = []
    for pair, data in data_dict.items():
        try:
            data = calculate_rsi(data)
            data = calculate_macd(data)

            last_rsi = data["RSI"].iloc[-1]
            last_macd = data["MACD"].iloc[-1]
            last_signal = data["Signal"].iloc[-1]
            last_volume = data["volume"].iloc[-1]
            average_volume = data["volume"].mean()
            last_close = data["close"].iloc[-1]

            trend = "восходящий" if data["close"].iloc[-1] > data["close"].iloc[-2] else "нисходящий"
            volume_status = "увеличен" if last_volume > average_volume * volume_threshold else "нормальный"

            recommendation = (
                f"{pair}: Тренд {trend}, объём {volume_status}, RSI: {last_rsi:.2f}, MACD: {last_macd:.2f}. "
                f"Цена закрытия: ${last_close:.2f}. "
            )
            if last_rsi < rsi_thresholds[0]:
                recommendation += "RSI показывает перепроданность. Рассмотрите покупку."
            elif last_rsi > rsi_thresholds[1]:
                recommendation += "RSI показывает перекупленность. Рассмотрите продажу."
            recommendations.append(recommendation)
        except Exception as e:
            recommendations.append(f"{pair}: Ошибка в расчётах - {str(e)}")
    return recommendations

def calculate_var(data, confidence_level=0.95):
    """
    Рассчитать Value at Risk (VaR) для заданного уровня доверия.

    Параметры:
    - data (DataFrame): Данные, содержащие цены закрытия.
    - confidence_level (float): Уровень доверия для расчета VaR (по умолчанию 95%).

    Возвращает:
    - VaR (float): Значение риска.
    """
    # Проверяем, есть ли колонка 'daily_return'. Если нет, рассчитываем ее.
    if 'daily_return' not in data.columns:
        if 'close' not in data.columns:
            raise KeyError("Данные не содержат колонку 'close' для расчета 'daily_return'.")
        data['daily_return'] = data['close'].pct_change()

    # Рассчитываем VaR
    return np.percentile(data['daily_return'].dropna(), (1 - confidence_level) * 100)

def calculate_atr(data, window=14):
    """
    Рассчитать Average True Range (ATR) для заданного периода.

    Параметры:
    - data (DataFrame): Данные с колонками 'high', 'low' и 'close'.
    - window (int): Период для расчета ATR (по умолчанию 14).

    Возвращает:
    - DataFrame: Обновленный DataFrame с колонкой 'ATR'.
    """
    if not {'high', 'low', 'close'}.issubset(data.columns):
        raise KeyError("Для расчета ATR необходимы колонки 'high', 'low' и 'close'.")

    data['TR'] = data[['high', 'low', 'close']].apply(
        lambda row: max(
            row['high'] - row['low'],
            abs(row['high'] - row['close']),
            abs(row['low'] - row['close'])
        ), axis=1
    )

    data['ATR'] = data['TR'].rolling(window=window).mean()
    return data

def calculate_bollinger_bands(data, window=20):
    """
    Рассчитывает полосы Боллинджера (Bollinger Bands) для набора данных.

    Параметры:
    - data (DataFrame): Исторические данные, должны содержать колонку 'close'.
    - window (int): Период для расчета скользящего среднего и стандартного отклонения.

    Возвращает:
    - DataFrame с добавленными колонками 'BB_upper', 'BB_middle', 'BB_lower'.
    """
    if 'close' not in data.columns:
        raise KeyError("Данные должны содержать колонку 'close' для расчета полос Боллинджера.")

    data['BB_middle'] = data['close'].rolling(window=window).mean()
    data['BB_std'] = data['close'].rolling(window=window).std()
    data['BB_upper'] = data['BB_middle'] + (2 * data['BB_std'])
    data['BB_lower'] = data['BB_middle'] - (2 * data['BB_std'])

    # Удаляем временную колонку
    data.drop(columns=['BB_std'], inplace=True)

    return data

def calculate_correlation_matrix(data_dict):
    """
    Рассчитывает корреляционную матрицу для данных нескольких активов.

    Параметры:
    - data_dict (dict): Словарь, где ключи — активы, значения — DataFrame с колонкой 'close'.

    Возвращает:
    - DataFrame: Корреляционная матрица.
    """
    correlation_data = {}
    for asset, data in data_dict.items():
        if 'close' in data.columns:
            correlation_data[asset] = data['close'].pct_change().dropna()  # Вычисляем дневные изменения

    if correlation_data:
        correlation_df = pd.DataFrame(correlation_data)
        correlation_matrix = correlation_df.corr()  # Рассчитываем корреляцию
        return correlation_matrix
    else:
        raise ValueError("Нет доступных данных для расчета корреляционной матрицы.")

def monte_carlo_simulation(data, num_simulations=1000, num_days=30):
    """
    Выполняет моделирование методом Монте-Карло для прогнозирования цен.

    Параметры:
    - data (DataFrame): Исторические данные, включая цены закрытия.
    - num_simulations (int): Количество симуляций.
    - num_days (int): Количество дней для прогнозирования.

    Возвращает:
    - simulated_prices (DataFrame): Симулированные цены для каждого сценария.
    """
    if "close" not in data:
        raise ValueError("В данных отсутствует колонка 'close' для выполнения моделирования.")

    # Ежедневные изменения цен
    daily_returns = data["close"].pct_change().dropna()
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()

    # Матрица для хранения симулированных цен
    last_price = data["close"].iloc[-1]
    simulated_prices = np.zeros((num_days, num_simulations))

    # Выполняем моделирование
    for simulation in range(num_simulations):
        prices = [last_price]
        for day in range(1, num_days):
            price = prices[-1] * (1 + np.random.normal(mean_return, std_dev))
            prices.append(price)
        simulated_prices[:, simulation] = prices

    return pd.DataFrame(simulated_prices)

def bayesian_update(prior, likelihood, evidence):
    """
    Обновляет апостериорную вероятность с использованием Байесовской теоремы.

    Параметры:
    - prior (float): Априорная вероятность (до получения новых данных).
    - likelihood (float): Вероятность наблюдаемых данных при гипотезе.
    - evidence (float): Вероятность наблюдаемых данных.

    Возвращает:
    - posterior (float): Апостериорная вероятность (после учета новых данных).
    """
    posterior = (likelihood * prior) / evidence
    return posterior

def calculate_bayes_laplace(data, probabilities):
    """
    Рассчитывает критерий Байеса-Лапласа для набора данных.

    Параметры:
    - data (DataFrame): Матрица выплат, где строки - альтернативы, а столбцы - сценарии.
    - probabilities (list): Список вероятностей для каждого сценария.

    Возвращает:
    - (Series): Значения критерия Байеса-Лапласа для каждой альтернативы.
    """
    if len(probabilities) != data.shape[1]:
        raise ValueError("Количество вероятностей должно совпадать с количеством сценариев.")
    return data.dot(probabilities)

def calculate_savage(data):
    """
    Рассчитывает критерий Сэвиджа для набора данных.

    Параметры:
    - data (DataFrame): Матрица выплат, где строки - альтернативы, а столбцы - сценарии.

    Возвращает:
    - (Series): Значения критерия Сэвиджа для каждой альтернативы.
    """
    regret_matrix = data.max(axis=0) - data
    return regret_matrix.max(axis=1)

def calculate_hurwicz(data, alpha=0.5):
    """
    Рассчитывает критерий Гурвица для набора данных.

    Параметры:
    - data (DataFrame): Матрица выплат, где строки - альтернативы, а столбцы - сценарии.
    - alpha (float): Коэффициент оптимизма (значение от 0 до 1).

    Возвращает:
    - (Series): Значения критерия Гурвица для каждой альтернативы.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Коэффициент оптимизма должен быть в диапазоне от 0 до 1.")
    return alpha * data.max(axis=1) + (1 - alpha) * data.min(axis=1)

def calculate_sharpe_ratio(data, risk_free_rate=0.01):
    """
    Расчет коэффициента Шарпа для оценки риска и доходности.

    :param data: DataFrame с колонкой 'daily_return'.
    :param risk_free_rate: Безрисковая ставка доходности (по умолчанию 1%).
    :return: Sharpe Ratio.
    """
    if 'daily_return' not in data:
        data['daily_return'] = data['close'].pct_change()  # Расчет дневной доходности
    excess_return = data['daily_return'].mean() - risk_free_rate
    return excess_return / data['daily_return'].std()

def generate_trend_recommendations(data):
    """
    Генерация рекомендаций на основе трендов SMA и EMA.

    :param data: DataFrame с колонками 'SMA' и 'EMA'.
    :return: Список рекомендаций.
    """
    recommendations = []
    if data['EMA'].iloc[-1] > data['SMA'].iloc[-1]:
        recommendations.append("Текущий тренд восходящий. Рекомендуется покупка.")
    else:
        recommendations.append("Текущий тренд нисходящий. Рассмотрите продажу.")
    return recommendations

def highlight_risk_zones(data, title="Зоны риска"):
    """
    Создает график с выделенными зонами высокого и низкого риска.
    """
    fig = go.Figure()

    # Основной график цен
    fig.add_trace(go.Scatter(
        x=data["open_time"],
        y=data["close"],
        mode="lines",
        name="Цена",
        line=dict(color="blue")
    ))

    # Маска для зоны высокого риска
    high_risk_mask = data["volatility"] > data["volatility"].mean()

    # Зона высокого риска
    fig.add_trace(go.Scatter(
        x=data["open_time"][high_risk_mask],
        y=data["close"][high_risk_mask],
        mode="markers",
        marker=dict(color="red", size=5),
        name="Высокий риск"
    ))

    # Зона низкого риска
    fig.add_trace(go.Scatter(
        x=data["open_time"][~high_risk_mask],
        y=data["close"][~high_risk_mask],
        mode="markers",
        marker=dict(color="green", size=5),
        name="Низкий риск"
    ))

    # Настройка графика
    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Цена",
        template="plotly_white"
    )

    return fig

def detect_trends(data):
    """
    Определяет восходящие и нисходящие тренды на основе EMA.
    Возвращает DataFrame с колонкой 'trend' (up/down).
    """
    data["trend"] = "stable"
    data.loc[data["close"] > data["EMA"], "trend"] = "up"
    data.loc[data["close"] < data["EMA"], "trend"] = "down"
    return data

def calculate_bayesian_probabilities(data, threshold=0.01):
    """
    Рассчитывает апостериорные вероятности для движения цены вверх/вниз.

    Параметры:
    - data (DataFrame): Исторические данные с колонкой "close".
    - threshold (float): Порог для определения значимого изменения цены.

    Возвращает:
    - probabilities (dict): Апостериорные вероятности для гипотез.
    """
    if "close" not in data:
        raise ValueError("В данных отсутствует колонка 'close'.")

    # Рассчитываем изменения цен
    data['price_change'] = data['close'].pct_change()

    # Определяем априорные вероятности
    prior_up = 0.5
    prior_down = 0.5

    # Вероятности при изменении цены выше/ниже порога
    likelihood_up = np.mean(data['price_change'] > threshold)
    likelihood_down = np.mean(data['price_change'] < -threshold)

    # Общая вероятность изменения
    evidence = likelihood_up + likelihood_down

    # Обновляем вероятности
    posterior_up = bayesian_update(prior_up, likelihood_up, evidence)
    posterior_down = bayesian_update(prior_down, likelihood_down, evidence)

    return {"up": posterior_up, "down": posterior_down}