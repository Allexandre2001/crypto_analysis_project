import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def preprocess_data(data, interval="1m"):
    """Попередня обробка даних для підвищення продуктивності."""
    if interval == "1m":
        data = data.resample("5T", on="open_time").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna().reset_index()
    return data

def plot_price_and_volume_optimized(data, interval="1m", sma_window=14, ema_window=14, chart_type="line"):
    """
    Оптимізоване побудова графіків для цін і обсягів із синхронізацією.
    """
    # Агрегація даних, якщо необхідно
    if interval == "1m":
        data = preprocess_data(data, interval)

    # Створюємо графік з двома рядами
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],  # Верхній графік для цін (70%), нижній для обсягів (30%)
        vertical_spacing=0.03
    )

    # Верхній графік - ціни
    if chart_type == "line":
        fig.add_trace(go.Scatter(x=data["open_time"], y=data["close"], mode="lines", name="Ціна закриття"), row=1, col=1)
    elif chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=data["open_time"], open=data["open"], high=data["high"], low=data["low"], close=data["close"],
            name="Свічковий графік"), row=1, col=1
        )
    elif chart_type == "bar":
        fig.add_trace(go.Bar(x=data["open_time"], y=data["close"], name="Стовпчиковий графік"), row=1, col=1)

    # Додаємо індикатори на графік цін
    data["SMA"] = data["close"].rolling(window=sma_window).mean()
    data["EMA"] = data["close"].ewm(span=ema_window, adjust=False).mean()
    fig.add_trace(go.Scatter(x=data["open_time"], y=data["SMA"], mode="lines", name=f"SMA ({sma_window})"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["open_time"], y=data["EMA"], mode="lines", name=f"EMA ({ema_window})"), row=1, col=1)

    # Нижній графік - обсяги
    fig.add_trace(go.Bar(x=data["open_time"], y=data["volume"], name="Обсяг"), row=2, col=1)

    # Налаштування осей
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Ціна (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Обсяг", row=2, col=1)

    # Загальні налаштування
    fig.update_layout(
        title="Графіки цін і обсягів",
        template="plotly_white",
        dragmode="pan",  # Панорамування
        xaxis=dict(
            rangeslider=dict(visible=False)  # Синхронізація осі X
        )
    )

    return fig

def plot_comparison(data_long, data_short, title="Порівняльний аналіз"):
    """
    Побудова порівняльного графіка довгострокових і короткострокових даних.

    Параметри:
    - data_long (DataFrame): Довгострокові дані.
    - data_short (DataFrame): Короткострокові дані.
    - title (str): Заголовок графіка.

    Повертає:
    - fig (Figure): Об'єкт Plotly Figure.
    """
    fig = go.Figure()

    # Довгостроковий графік
    fig.add_trace(go.Scatter(
        x=data_long["open_time"], y=data_long["close"],
        mode="lines", name="Довгостроковий тренд", line=dict(color="blue")
    ))

    # Короткостроковий графік
    fig.add_trace(go.Scatter(
        x=data_short["open_time"], y=data_short["close"],
        mode="lines", name="Короткостроковий тренд", line=dict(color="orange")
    ))

    # Налаштування графіка
    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Ціна (USD)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def plot_bollinger_bands(data, title="Смуги Боллінджера"):
    """
    Побудова графіка смуг Боллінджера за допомогою Plotly.

    Параметри:
    - data (DataFrame): Дані з колонками 'close', 'BB_upper', 'BB_middle', 'BB_lower'.
    - title (str): Заголовок графіка.

    Повертає:
    - Об'єкт Plotly Figure.
    """
    fig = go.Figure()

    # Графік закриття цін
    fig.add_trace(go.Scatter(
        x=data['open_time'], y=data['close'],
        mode='lines', name='Ціна закриття', line=dict(color='blue')
    ))

    # Верхня смуга
    fig.add_trace(go.Scatter(
        x=data['open_time'], y=data['BB_upper'],
        mode='lines', name='BB Upper', line=dict(color='red'), opacity=0.5
    ))

    # Середня смуга
    fig.add_trace(go.Scatter(
        x=data['open_time'], y=data['BB_middle'],
        mode='lines', name='BB Middle', line=dict(color='green'), opacity=0.5
    ))

    # Нижня смуга
    fig.add_trace(go.Scatter(
        x=data['open_time'], y=data['BB_lower'],
        mode='lines', name='BB Lower', line=dict(color='red'), opacity=0.5
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Ціна (USD)",
        template="plotly_white"
    )

    return fig

def plot_correlation_matrix(correlation_matrix, title="Кореляційна матриця"):
    """
    Візуалізація кореляційної матриці.

    Параметри:
    - correlation_matrix (DataFrame): Кореляційна матриця.
    - title (str): Заголовок графіка.

    Повертає:
    - Об'єкт Plotly Figure.
    """
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale="Viridis",
        title=title
    )
    fig.update_layout(
        xaxis_title="Активи",
        yaxis_title="Активи",
        template="plotly_white"
    )
    return fig

def plot_monte_carlo(simulated_prices, title="Моделювання методом Монте-Карло"):
    """
    Побудова графіка результатів моделювання методом Монте-Карло.

    Параметри:
    - simulated_prices (DataFrame): Симульовані ціни для кожного сценарію.
    - title (str): Заголовок графіка.

    Повертає:
    - fig (Figure): Графік Plotly.
    """
    fig = go.Figure()

    # Додаємо всі сценарії
    for col in simulated_prices.columns:
        fig.add_trace(go.Scatter(
            x=simulated_prices.index,
            y=simulated_prices[col],
            mode="lines",
            line=dict(width=0.5, color="blue"),
            opacity=0.1,
            showlegend=False
        ))

    # Середній сценарій
    fig.add_trace(go.Scatter(
        x=simulated_prices.index,
        y=simulated_prices.mean(axis=1),
        mode="lines",
        line=dict(width=2, color="red"),
        name="Середній сценарій"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Дні",
        yaxis_title="Ціна (USD)",
        template="plotly_white"
    )
    return fig

def plot_criteria_results(criteria_results, title="Ризик і дохідність"):
    """
    Візуалізація результатів критеріїв Байєса-Лапласа, Севіджа та Гурвіца.

    Аргументи:
    - criteria_results: словник із результатами критеріїв.
    - title: заголовок графіка.

    Повертає:
    - fig: інтерактивний графік.
    """
    fig = go.Figure()

    for criterion, values in criteria_results.items():
        fig.add_trace(go.Bar(
            x=list(values.keys()),
            y=list(values.values()),
            name=criterion
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Альтернативи",
        yaxis_title="Значення",
        barmode="group",
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            title="Альтернативи",
            tickangle=45,
            tickfont=dict(size=10),
            fixedrange=False  # Вимкнення фіксованого масштабу
        ),
        yaxis=dict(
            showgrid=True,
            title="Значення",
            fixedrange=False  # Вимкнення фіксованого масштабу
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def plot_long_short(data_long, data_short, pair):
    """
    Побудова графіка довгострокових і короткострокових даних із анотаціями.

    Параметри:
    - data_long: DataFrame довгострокових даних.
    - data_short: DataFrame короткострокових даних.
    - pair: Назва пари.

    Повертає:
    - fig: Об'єкт Plotly Figure.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Довгостроковий тренд
    fig.add_trace(
        go.Scatter(
            x=data_long["open_time"],
            y=data_long["close"],
            mode="lines",
            name="Довгостроковий тренд",
            line=dict(color="blue")
        ),
        secondary_y=False,
    )

    # Короткостроковий тренд
    fig.add_trace(
        go.Scatter(
            x=data_short["open_time"],
            y=data_short["close"],
            mode="lines",
            name="Короткостроковий тренд",
            line=dict(color="orange")
        ),
        secondary_y=True,
    )

    # Налаштування осей
    fig.update_xaxes(title_text="Дата")
    fig.update_yaxes(title_text="Довгострокова ціна", secondary_y=False)
    fig.update_yaxes(title_text="Короткострокова ціна", secondary_y=True)

    # Налаштування графіка
    fig.update_layout(
        title=f"Довгострокові та короткострокові тренди для {pair}",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def plot_bayesian_probabilities(probabilities, title="Байєсівські ймовірності"):
    """
    Візуалізація байєсівських ймовірностей.

    :param probabilities: Словник ймовірностей {'label': value}.
    :param title: Заголовок графіка.
    :return: Об'єкт Plotly Figure.
    """
    # Розділення словника на мітки та значення
    labels = list(probabilities.keys())
    values = list(probabilities.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        name="Ймовірності",
        marker_color="blue"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Категорії",
        yaxis_title="Ймовірність",
        template="plotly_white",
        showlegend=False
    )

    return fig

def plot_criteria_results(criteria_results, title="Результати оцінки"):
    """
    Побудова графіка результатів різних критеріїв.

    Параметри:
    - criteria_results (dict): Словник, де ключі - назви критеріїв, значення - результати (Series).
    - title (str): Заголовок графіка.

    Повертає:
    - fig (Figure): Об'єкт графіка Plotly.
    """
    fig = go.Figure()

    for criterion, results in criteria_results.items():
        fig.add_trace(go.Bar(
            x=results.index,
            y=results.values,
            name=criterion
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Альтернативи",
        yaxis_title="Значення",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_risk_zones(data, atr_threshold):
    """
    Побудова графіка із зонами ризику на основі ATR.

    :param data: DataFrame із даними.
    :param atr_threshold: Порогове значення ATR для визначення зон ризику.
    :return: Графік Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['open_time'], y=data['close'], mode='lines', name='Ціна'))

    # Додаємо зони ризику
    high_risk = data['ATR'] > atr_threshold
    fig.add_trace(go.Scatter(
        x=data.loc[high_risk, 'open_time'],
        y=data.loc[high_risk, 'close'],
        mode='markers',
        marker=dict(color='red', size=6),
        name='Високий ризик'
    ))

    fig.update_layout(
        title="Зони ризику на основі ATR",
        xaxis_title="Дата",
        yaxis_title="Ціна",
        template="plotly_white"
    )
    return fig

def plot_trends(data, title="Аналіз трендів"):
    """
    Відображає графік цін із кольоровим виділенням трендів.
    """
    fig = go.Figure()

    # Висхідний тренд
    up_trend = data[data["trend"] == "up"]
    fig.add_trace(go.Scatter(
        x=up_trend["open_time"],
        y=up_trend["close"],
        mode="lines",
        line=dict(color="green"),
        name="Висхідний тренд"
    ))

    # Низхідний тренд
    down_trend = data[data["trend"] == "down"]
    fig.add_trace(go.Scatter(
        x=down_trend["open_time"],
        y=down_trend["close"],
        mode="lines",
        line=dict(color="red"),
        name="Низхідний тренд"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Ціна",
        template="plotly_white"
    )
    return fig

def display_table(data, title="Таблиця даних"):
    """Відображення таблиці даних."""
    if data.empty:
        st.warning("Дані відсутні.")
        return
    st.subheader(title)
    st.dataframe(data)
