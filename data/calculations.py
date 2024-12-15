import numpy as np
import pandas as pd
import plotly.graph_objects as go

def clean_data(df):
    """
     Видаляє непотрібні колонки і залишає лише ключові: open_time, open, high, low, close, volume.

     Параметри:
     df (pd.DataFrame): Початковий DataFrame з даними активу.

     Повертає:
     pd.DataFrame: Очищений DataFrame.
     """
    columns_to_keep = ["open_time", "open", "high", "low", "close", "volume"]
    return df[columns_to_keep]

def calculate_returns(df):
    """
    Обчислення щоденного повернення активу.

    Аргументи:
    - data: DataFrame з ціновими даними.

    Повертає:
    - DataFrame з доданим стовпцем 'daily_return'.
     """
    if df.empty or "close" not in df:
        raise ValueError("Дані відсутні або не містять колонки 'close'.")
    df["daily_return"] = df["close"].pct_change()
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df

def calculate_volatility(df):
    """
 Розраховує волатильність на основі стандартного відхилення денної доходності.

 Параметри:
 df (DataFrame): Дані активу з колонкою "daily_return".

 Повертає:
 float: Значення волатильності.
 """
    if df.empty or "daily_return" not in df:
        raise ValueError("Дані відсутні або не містять колонки 'daily_return'.")
    return df["daily_return"].std()

def calculate_correlation(data_dict):
    """
     Розраховує кореляцію між активами.

     Параметри:
     data_dict (dict): Словник із парами активів та їх даними.

     Повертає:
     DataFrame: Матриця кореляції.
     """
    close_prices = {pair: data["close"] for pair, data in data_dict.items()}
    close_df = pd.DataFrame(close_prices)
    return close_df.corr()

def calculate_moving_averages(df, window=14):
    """
    Розраховує ковзні середні (SMA, EMA).

    Параметри:
    df (DataFrame): Дані активу з колонкою 'close'.
    window (int): Період для розрахунку SMA та EMA.

    Повертає:
    DataFrame: Оновлений DataFrame з колонками 'SMA' та 'EMA'.
    """
    if 'close' not in df.columns:
        raise KeyError("Колонка 'close' відсутня у даних.")

    # Розрахунок SMA
    df['SMA'] = df['close'].rolling(window=window).mean()
    # Розрахунок EMA
    df['EMA'] = df['close'].ewm(span=window, adjust=False).mean()

    return df

def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    """
    Розраховує коефіцієнт Sharpe для оцінки співвідношення дохідності і ризику.

    Параметри:
    df (DataFrame): Дані активу з колонкою "daily_return".
    risk_free_rate (float): Безризикова ставка.

    Повертає:
    float: Значення коефіцієнта Sharpe.
    """
    if df.empty or "daily_return" not in df:
        raise ValueError("Дані відсутні або не містять колонку 'daily_return'.")
    mean_return = df["daily_return"].mean()
    risk = df["daily_return"].std()
    return (mean_return - risk_free_rate) / risk if risk != 0 else np.nan

def calculate_rsi(df, window=14):
    """
    Розраховує RSI (Relative Strength Index).

    Параметри:
    df (DataFrame): Дані активу з колонкою "close".
    window (int): Період для розрахунку RSI.

    Повертає:
    DataFrame: Оновлений DataFrame з колонкою "RSI".
    """
    if df.empty or "close" not in df:
        raise ValueError("Дані відсутні або не містять колонку 'close'.")
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
    Розраховує MACD і сигнальну лінію.

    Параметри:
    df (DataFrame): Дані активу з колонкою "close".
    short_window (int): Період для швидкого EMA.
    long_window (int): Період для повільного EMA.
    signal_window (int): Період для сигнальної лінії.

    Повертає:
    DataFrame: Оновлений DataFrame з колонками "MACD" та "Signal".
    """
    if df.empty or "close" not in df:
        raise ValueError("Дані відсутні або не містять колонку 'close'.")
    df["MACD"] = df["close"].ewm(span=short_window, adjust=False).mean() - \
                 df["close"].ewm(span=long_window, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    return df

def generate_advanced_recommendations(data_dict, rsi_thresholds=(30, 70), volume_threshold=1.5):
    """
    Генерує розширені рекомендації на основі MACD, RSI, обсягів та трендів.

    Параметри:
    data_dict (dict): Словник із парами активів та їх даними.
    rsi_thresholds (tuple): Порогові значення RSI для перепроданості та перекупленості.
    volume_threshold (float): Коефіцієнт для аналізу обсягу.

    Повертає:
    list: Список рекомендацій.
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

            trend = "висхідний" if data["close"].iloc[-1] > data["close"].iloc[-2] else "низхідний"
            volume_status = "збільшений" if last_volume > average_volume * volume_threshold else "нормальний"

            recommendation = (
                f"{pair}: Тренд {trend}, обсяг {volume_status}, RSI: {last_rsi:.2f}, MACD: {last_macd:.2f}. "
                f"Ціна закриття: ${last_close:.2f}. "
            )
            if last_rsi < rsi_thresholds[0]:
                recommendation += "RSI показує перепроданість. Розгляньте покупку."
            elif last_rsi > rsi_thresholds[1]:
                recommendation += "RSI показує перекупленість. Розгляньте продаж."
            recommendations.append(recommendation)
        except Exception as e:
            recommendations.append(f"{pair}: Помилка у розрахунках - {str(e)}")
    return recommendations

def calculate_var(data, confidence_level=0.95):
    """
    Розраховує Value at Risk (VaR) для заданого рівня довіри.

    Параметри:
    - data (DataFrame): Дані, що містять ціни закриття.
    - confidence_level (float): Рівень довіри для розрахунку VaR (за замовчуванням 95%).

    Повертає:
    - VaR (float): Значення ризику.
    """
    if 'daily_return' not in data.columns:
        if 'close' not in data.columns:
            raise KeyError("Дані не містять колонку 'close' для розрахунку 'daily_return'.")
        data['daily_return'] = data['close'].pct_change()

    return np.percentile(data['daily_return'].dropna(), (1 - confidence_level) * 100)

def calculate_atr(data, window=14):
    """
    Розраховує Average True Range (ATR) для заданого періоду.

    Параметри:
    - data (DataFrame): Дані з колонками 'high', 'low' та 'close'.
    - window (int): Період для розрахунку ATR (за замовчуванням 14).

    Повертає:
    - DataFrame: Оновлений DataFrame з колонкою 'ATR'.
    """
    if not {'high', 'low', 'close'}.issubset(data.columns):
        raise KeyError("Для розрахунку ATR потрібні колонки 'high', 'low' та 'close'.")

    data['TR'] = data[['high', 'low', 'close']].apply(
        lambda row: max(
            row['high'] - row['low'],
            abs(row['high'] - row['close']),
            abs(row['low'] - row['close'])
        ), axis=1
    )

    data['ATR'] = data['TR'].rolling(window=window).mean()
    return data

def process_indicators(data_dict, pair, atr_window=14, bollinger_window=20):
    """
    Розрахунок індикаторів для заданої криптовалютної пари та оновлення даних.

    Параметри:
    data_dict (dict): Словник із даними для кожної пари.
    pair (str): Назва пари (наприклад, BTCUSDT).
    atr_window (int): Період для розрахунку ATR.
    bollinger_window (int): Період для смуг Боллінджера.

    Повертає:
    dict: Оновлені дані з розрахованими індикаторами.
    """
    data = data_dict.get(pair)

    if data is None or data.empty:
        raise ValueError(f"Дані для {pair} відсутні або порожні.")

    columns_needed = ["open_time", "open", "high", "low", "close", "volume"]
    data = data[columns_needed]

    if "ATR" not in data.columns:
        data = calculate_atr(data, window=atr_window)

    if "daily_return" not in data.columns:
        data["daily_return"] = data["close"].pct_change()
    var_95 = calculate_var(data)

    sharpe_ratio = calculate_sharpe_ratio(data)

    if not {"BB_upper", "BB_middle", "BB_lower"}.issubset(data.columns):
        data = calculate_bollinger_bands(data, window=bollinger_window)

    data_dict[pair] = data

    return {
        "ATR_mean": data["ATR"].mean(),
        "VaR_95": var_95,
        "Sharpe_Ratio": sharpe_ratio,
        "Bollinger_Bands": {
            "upper": data["BB_upper"].iloc[-1],
            "middle": data["BB_middle"].iloc[-1],
            "lower": data["BB_lower"].iloc[-1]
        }
    }



def calculate_bollinger_bands(data, window=20):
    """
    Розраховує смуги Боллінджера (Bollinger Bands) для набору даних.

    Параметри:
    - data (DataFrame): Історичні дані, повинні містити колонку 'close'.
    - window (int): Період для розрахунку ковзного середнього та стандартного відхилення.

    Повертає:
    - DataFrame з доданими колонками 'BB_upper', 'BB_middle', 'BB_lower'.
    """
    if 'close' not in data.columns:
        raise KeyError("Дані повинні містити колонку 'close' для розрахунку смуг Боллінджера.")

    data['BB_middle'] = data['close'].rolling(window=window).mean()
    data['BB_std'] = data['close'].rolling(window=window).std()
    data['BB_upper'] = data['BB_middle'] + (2 * data['BB_std'])
    data['BB_lower'] = data['BB_middle'] - (2 * data['BB_std'])

    data.drop(columns=['BB_std'], inplace=True)

    return data

def calculate_correlation_matrix(data_dict):
    """
    Розраховує кореляційну матрицю для даних кількох активів.

    Параметри:
    - data_dict (dict): Словник, де ключі — активи, значення — DataFrame із колонкою 'close'.

    Повертає:
    - DataFrame: Кореляційна матриця.
    """
    correlation_data = {}
    for asset, data in data_dict.items():
        if 'close' in data.columns:
            correlation_data[asset] = data['close'].pct_change().dropna()

    if correlation_data:
        correlation_df = pd.DataFrame(correlation_data)
        correlation_matrix = correlation_df.corr()
        return correlation_matrix
    else:
        raise ValueError("Немає доступних даних для розрахунку кореляційної матриці.")

def monte_carlo_simulation(data, num_simulations=1000, num_days=30):
    """
    Виконує моделювання методом Монте-Карло для прогнозування цін.

    Параметри:
    - data (DataFrame): Історичні дані, включаючи ціни закриття.
    - num_simulations (int): Кількість симуляцій.
    - num_days (int): Кількість днів для прогнозування.

    Повертає:
    - simulated_prices (DataFrame): Симульовані ціни для кожного сценарію.
    """
    if "close" not in data:
        raise ValueError("В даних відсутня колонка 'close' для виконання моделювання.")

    # Щоденні зміни цін
    daily_returns = data["close"].pct_change().dropna()
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()

    # Матриця для збереження симульованих цін
    last_price = data["close"].iloc[-1]
    simulated_prices = np.zeros((num_days, num_simulations))

    # Виконуємо моделювання
    for simulation in range(num_simulations):
        prices = [last_price]
        for day in range(1, num_days):
            price = prices[-1] * (1 + np.random.normal(mean_return, std_dev))
            prices.append(price)
        simulated_prices[:, simulation] = prices

    return pd.DataFrame(simulated_prices)

def bayesian_update(prior, likelihood, evidence):
    """
    Оновлює апостеріорну ймовірність за допомогою Байєсівської теореми.

    Параметри:
    - prior (float): Апріорна ймовірність (до отримання нових даних).
    - likelihood (float): Ймовірність спостережуваних даних за гіпотезою.
    - evidence (float): Ймовірність спостережуваних даних.

    Повертає:
    - posterior (float): Апостеріорна ймовірність (після врахування нових даних).
    """
    posterior = (likelihood * prior) / evidence
    return posterior

def calculate_bayes_laplace(data, probabilities):
    """
    Розраховує критерій Байєса-Лапласа для набору даних.

    Параметри:
    - data (DataFrame): Матриця виплат, де рядки - альтернативи, а стовпці - сценарії.
    - probabilities (list): Список ймовірностей для кожного сценарію.

    Повертає:
    - (Series): Значення критерію Байєса-Лапласа для кожної альтернативи.
    """
    if len(probabilities) != data.shape[1]:
        raise ValueError("Кількість ймовірностей повинна збігатися з кількістю сценаріїв.")
    return data.dot(probabilities)

def calculate_savage(data):
    """
    Розраховує критерій Севіджа для набору даних.

    Параметри:
    - data (DataFrame): Матриця виплат, де рядки - альтернативи, а стовпці - сценарії.

    Повертає:
    - (Series): Значення критерію Севіджа для кожної альтернативи.
    """
    regret_matrix = data.max(axis=0) - data
    return regret_matrix.max(axis=1)

def calculate_hurwicz(data, alpha=0.5):
    """
    Розраховує критерій Гурвіца для набору даних.

    Параметри:
    - data (DataFrame): Матриця виплат, де рядки - альтернативи, а стовпці - сценарії.
    - alpha (float): Коефіцієнт оптимізму (значення від 0 до 1).

    Повертає:
    - (Series): Значення критерію Гурвіца для кожної альтернативи.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Коефіцієнт оптимізму має бути в діапазоні від 0 до 1.")
    return alpha * data.max(axis=1) + (1 - alpha) * data.min(axis=1)

def calculate_sharpe_ratio(data, risk_free_rate=0.01):
    """
    Розрахунок коефіцієнта Шарпа для оцінки ризику і дохідності.

    :param data: DataFrame з колонкою 'daily_return'.
    :param risk_free_rate: Безризикова ставка дохідності (за замовчуванням 1%).
    :return: Sharpe Ratio.
    """
    if 'daily_return' not in data:
        data['daily_return'] = data['close'].pct_change()  # Розрахунок щоденної дохідності
    excess_return = data['daily_return'].mean() - risk_free_rate
    return excess_return / data['daily_return'].std()

def generate_trend_recommendations(data):
    """
    Генерація рекомендацій на основі трендів SMA та EMA.

    :param data: DataFrame з колонками 'SMA' і 'EMA'.
    :return: Список рекомендацій.
    """
    recommendations = []

    # Перевірка наявності колонок 'EMA' і 'SMA'
    if 'EMA' not in data.columns or 'SMA' not in data.columns:
        # Розраховуємо EMA і SMA, якщо їх немає
        data = calculate_moving_averages(data, window=14)  # Період можна налаштувати

    # Генерація рекомендацій
    if data['EMA'].iloc[-1] > data['SMA'].iloc[-1]:
        recommendations.append("Поточний тренд висхідний. Рекомендується купівля.")
    else:
        recommendations.append("Поточний тренд низхідний. Розгляньте продаж.")

    return recommendations

def highlight_risk_zones(data, title="Зони ризику"):
    """
    Створює графік із виділеними зонами високого та низького ризику.
    """
    fig = go.Figure()

    # Основний графік цін
    fig.add_trace(go.Scatter(
        x=data["open_time"],
        y=data["close"],
        mode="lines",
        name="Ціна",
        line=dict(color="blue")
    ))

    # Маска для зони високого ризику
    high_risk_mask = data["volatility"] > data["volatility"].mean()

    # Зона високого ризику
    fig.add_trace(go.Scatter(
        x=data["open_time"][high_risk_mask],
        y=data["close"][high_risk_mask],
        mode="markers",
        marker=dict(color="red", size=5),
        name="Високий ризик"
    ))

    # Зона низького ризику
    fig.add_trace(go.Scatter(
        x=data["open_time"][~high_risk_mask],
        y=data["close"][~high_risk_mask],
        mode="markers",
        marker=dict(color="green", size=5),
        name="Низький ризик"
    ))

    # Налаштування графіка
    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Ціна",
        template="plotly_white"
    )

    return fig

def detect_trends(data):
    """
    Визначає висхідні та низхідні тренди на основі EMA.
    Повертає DataFrame з колонкою 'trend' (up/down).
    """
    data["trend"] = "stable"
    data.loc[data["close"] > data["EMA"], "trend"] = "up"
    data.loc[data["close"] < data["EMA"], "trend"] = "down"
    return data

def calculate_bayesian_probabilities(data, threshold=0.01):
    """
    Розраховує апостеріорні ймовірності для руху ціни вгору/вниз.

    Параметри:
    - data (DataFrame): Історичні дані з колонкою "close".
    - threshold (float): Поріг для визначення значної зміни ціни.

    Повертає:
    - probabilities (dict): Апостеріорні ймовірності для гіпотез.
    """
    if "close" not in data:
        raise ValueError("В даних відсутня колонка 'close'.")

    # Розраховуємо зміни цін
    data['price_change'] = data['close'].pct_change()

    # Апріорні ймовірності
    prior_up = 0.5
    prior_down = 0.5

    # Ймовірності при зміні ціни вище/нижче порогу
    likelihood_up = np.mean(data['price_change'] > threshold)
    likelihood_down = np.mean(data['price_change'] < -threshold)

    # Загальна ймовірність зміни
    evidence = likelihood_up + likelihood_down

    # Оновлення ймовірностей
    posterior_up = bayesian_update(prior_up, likelihood_up, evidence)
    posterior_down = bayesian_update(prior_down, likelihood_down, evidence)

    return {"up": posterior_up, "down": posterior_down}