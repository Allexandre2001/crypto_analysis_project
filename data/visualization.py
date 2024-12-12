import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def preprocess_data(data, interval="1m"):
    """Предобработка данных для повышения производительности."""
    if interval == "1m":
        data = data.resample("5T", on="open_time").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna().reset_index()
    return data

def plot_price_and_volume_optimized(data, interval="1m", sma_window=14, ema_window=14, chart_type="line"):
    """
    Оптимизированное построение графиков для цен и объемов с синхронизацией.
    """
    # Агрегирование данных, если требуется
    if interval == "1m":
        data = preprocess_data(data, interval)

    # Создаем график с двумя рядами
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],  # Верхний график для цен (70%), нижний для объемов (30%)
        vertical_spacing=0.03
    )

    # Верхний график - цены
    if chart_type == "line":
        fig.add_trace(go.Scatter(x=data["open_time"], y=data["close"], mode="lines", name="Цена закрытия"), row=1, col=1)
    elif chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=data["open_time"], open=data["open"], high=data["high"], low=data["low"], close=data["close"],
            name="Свечной график"), row=1, col=1
        )
    elif chart_type == "bar":
        fig.add_trace(go.Bar(x=data["open_time"], y=data["close"], name="Баровый график"), row=1, col=1)

    # Добавляем индикаторы на график цен
    data["SMA"] = data["close"].rolling(window=sma_window).mean()
    data["EMA"] = data["close"].ewm(span=ema_window, adjust=False).mean()
    fig.add_trace(go.Scatter(x=data["open_time"], y=data["SMA"], mode="lines", name=f"SMA ({sma_window})"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["open_time"], y=data["EMA"], mode="lines", name=f"EMA ({ema_window})"), row=1, col=1)

    # Нижний график - объемы
    fig.add_trace(go.Bar(x=data["open_time"], y=data["volume"], name="Объем"), row=2, col=1)

    # Настройка осей
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Цена (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Объем", row=2, col=1)

    # Общие настройки
    fig.update_layout(
        title="Графики цен и объемов",
        template="plotly_white",
        dragmode="pan",  # Панорамирование
        xaxis=dict(
            rangeslider=dict(visible=False)  # Синхронизация оси X
        )
    )

    return fig

def plot_comparison(data_long, data_short, title="Сравнительный анализ"):
    """
    Построение сравнительного графика долгосрочных и краткосрочных данных.

    Параметры:
    - data_long (DataFrame): Долгосрочные данные.
    - data_short (DataFrame): Краткосрочные данные.
    - title (str): Заголовок графика.

    Возвращает:
    - fig (Figure): Объект Plotly Figure.
    """
    fig = go.Figure()

    # Долгосрочный график
    fig.add_trace(go.Scatter(
        x=data_long["open_time"], y=data_long["close"],
        mode="lines", name="Долгосрочный тренд", line=dict(color="blue")
    ))

    # Краткосрочный график
    fig.add_trace(go.Scatter(
        x=data_short["open_time"], y=data_short["close"],
        mode="lines", name="Краткосрочный тренд", line=dict(color="orange")
    ))

    # Настройка графика
    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Цена (USD)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def plot_bollinger_bands(data, title="Bollinger Bands"):
    """
    Построение графика полос Боллинджера с помощью Plotly.

    Параметры:
    - data (DataFrame): Данные с колонками 'close', 'BB_upper', 'BB_middle', 'BB_lower'.
    - title (str): Заголовок графика.

    Возвращает:
    - Объект Plotly Figure.
    """
    fig = go.Figure()

    # График закрытия цен
    fig.add_trace(go.Scatter(
        x=data['open_time'], y=data['close'],
        mode='lines', name='Цена закрытия', line=dict(color='blue')
    ))

    # Верхняя полоса
    fig.add_trace(go.Scatter(
        x=data['open_time'], y=data['BB_upper'],
        mode='lines', name='BB Upper', line=dict(color='red'), opacity=0.5
    ))

    # Средняя полоса
    fig.add_trace(go.Scatter(
        x=data['open_time'], y=data['BB_middle'],
        mode='lines', name='BB Middle', line=dict(color='green'), opacity=0.5
    ))

    # Нижняя полоса
    fig.add_trace(go.Scatter(
        x=data['open_time'], y=data['BB_lower'],
        mode='lines', name='BB Lower', line=dict(color='red'), opacity=0.5
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Цена (USD)",
        template="plotly_white"
    )

    return fig

def plot_correlation_matrix(correlation_matrix, title="Корреляционная матрица"):
    """
    Визуализация корреляционной матрицы.

    Параметры:
    - correlation_matrix (DataFrame): Корреляционная матрица.
    - title (str): Заголовок графика.

    Возвращает:
    - Объект Plotly Figure.
    """
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale="Viridis",
        title=title
    )
    fig.update_layout(
        xaxis_title="Активы",
        yaxis_title="Активы",
        template="plotly_white"
    )
    return fig

def plot_monte_carlo(simulated_prices, title="Моделирование методом Монте-Карло"):
    """
    Строит график результатов моделирования методом Монте-Карло.

    Параметры:
    - simulated_prices (DataFrame): Симулированные цены для каждого сценария.
    - title (str): Заголовок графика.

    Возвращает:
    - fig (Figure): График Plotly.
    """
    fig = go.Figure()

    # Добавляем все сценарии
    for col in simulated_prices.columns:
        fig.add_trace(go.Scatter(
            x=simulated_prices.index,
            y=simulated_prices[col],
            mode="lines",
            line=dict(width=0.5, color="blue"),
            opacity=0.1,
            showlegend=False
        ))

    # Средний сценарий
    fig.add_trace(go.Scatter(
        x=simulated_prices.index,
        y=simulated_prices.mean(axis=1),
        mode="lines",
        line=dict(width=2, color="red"),
        name="Средний сценарий"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Дни",
        yaxis_title="Цена (USD)",
        template="plotly_white"
    )
    return fig

def plot_bayesian_probabilities(probabilities, title="Байесовское обновление вероятностей"):
    """
    Строит диаграмму вероятностей.

    Параметры:
    - probabilities (dict): Вероятности (например, {"up": 0.6, "down": 0.4}).
    - title (str): Заголовок графика.

    Возвращает:
    - fig (Figure): График Plotly.
    """
    labels = list(probabilities.keys())
    values = list(probabilities.values())

    fig = px.pie(
        names=labels,
        values=values,
        title=title,
        color=labels,
        color_discrete_map={"up": "green", "down": "red"}
    )
    fig.update_traces(textinfo="percent+label")
    return fig

def plot_long_short(data_long, data_short, pair):
    """
    Построение графика долгосрочных и краткосрочных данных с аннотацией.

    Параметры:
    - data_long: DataFrame долгосрочных данных.
    - data_short: DataFrame краткосрочных данных.
    - pair: Название пары.

    Возвращает:
    - fig: Объект Plotly Figure.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Долгосрочный тренд
    fig.add_trace(
        go.Scatter(
            x=data_long["open_time"],
            y=data_long["close"],
            mode="lines",
            name="Долгосрочный тренд",
            line=dict(color="blue")
        ),
        secondary_y=False,
    )

    # Краткосрочный тренд
    fig.add_trace(
        go.Scatter(
            x=data_short["open_time"],
            y=data_short["close"],
            mode="lines",
            name="Краткосрочный тренд",
            line=dict(color="orange")
        ),
        secondary_y=True,
    )

    # Настройка осей
    fig.update_xaxes(title_text="Дата")
    fig.update_yaxes(title_text="Долгосрочная цена", secondary_y=False)
    fig.update_yaxes(title_text="Краткосрочная цена", secondary_y=True)

    # Настройки графика
    fig.update_layout(
        title=f"Долгосрочные и краткосрочные тренды для {pair}",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_criteria_results(criteria_results, title="Результаты оценки"):
    """
    Построение графика результатов различных критериев.

    Параметры:
    - criteria_results (dict): Словарь, где ключи - названия критериев, значения - результаты (Series).
    - title (str): Заголовок графика.

    Возвращает:
    - fig (Figure): Объект графика Plotly.
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
        xaxis_title="Альтернативы",
        yaxis_title="Значения",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def display_table(data, title="Таблица данных"):
    """Отображение таблицы данных."""
    if data.empty:
        st.warning("Данные отсутствуют.")
        return
    st.subheader(title)
    st.dataframe(data)
