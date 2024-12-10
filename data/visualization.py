import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_interactive_graph(data, title="График актива"):
    """
    Построение интерактивного графика с использованием Plotly.
    """
    fig = px.line(
        data,
        x="open_time",
        y="close",
        title=title,
        labels={"open_time": "Дата", "close": "Цена (USD)"}
    )
    fig.update_traces(line_color="#3498db", line_width=2)
    fig.update_layout(
        title_font_size=20,
        template="plotly_white",
        xaxis=dict(rangeslider=dict(visible=True), title="Дата"),
        yaxis=dict(title="Цена (USD)"),
        dragmode="pan"  # Включение масштабирования и перемещения
    )
    return fig

def plot_with_indicators(data, title="График с индикаторами", sma_window=14, ema_window=14, chart_type="line"):
    """
    Построение графика с индикаторами (SMA, EMA) и объёмами.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Создаем subplot: верхний график - цены и индикаторы, нижний график - объёмы
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],  # Верхний график занимает 70%, нижний - 30%
        vertical_spacing=0.05  # Уменьшаем расстояние между графиками
    )

    # Верхний график (цены)
    if chart_type == "line":
        fig.add_trace(
            go.Scatter(x=data["open_time"], y=data["close"], mode="lines", name="Цена закрытия"),
            row=1, col=1
        )
    elif chart_type == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data["open_time"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Свечной график"
            ),
            row=1, col=1
        )
    elif chart_type == "bar":
        fig.add_trace(
            go.Bar(x=data["open_time"], y=data["close"], name="Баровый график"),
            row=1, col=1
        )

    # Добавляем индикаторы
    fig.add_trace(
        go.Scatter(x=data["open_time"], y=data["SMA"], mode="lines", name=f"SMA ({sma_window})"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data["open_time"], y=data["EMA"], mode="lines", name=f"EMA ({ema_window})"),
        row=1, col=1
    )

    # Нижний график (объёмы)
    fig.add_trace(
        go.Bar(
            x=data["open_time"], y=data["volume"],
            name="Объём",
            marker_color="rgba(100, 149, 237, 0.6)"  # Светло-голубой с прозрачностью
        ),
        row=2, col=1
    )

    # Настройка осей
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Цена (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Объём", row=2, col=1)

    # Общие настройки графика
    fig.update_layout(
        title=title,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="pan",
        xaxis2=dict(
            rangeslider=dict(visible=False),  # Отключаем слайдер на нижнем графике
            showline=True,
        )
    )

    return fig

def display_table(data, title="Таблица данных"):
    """
    Отображение таблицы данных с помощью Streamlit.
    """
    st.subheader(title)
    st.dataframe(data)
