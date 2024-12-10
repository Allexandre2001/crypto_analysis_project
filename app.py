import streamlit as st
from data.api import get_historical_data
from data.calculations import calculate_returns, calculate_volatility, generate_advanced_recommendations
from data.visualization import plot_with_indicators, display_table
from data.calculations import calculate_moving_averages
import pandas as pd

def load_custom_css():
    """Загрузка пользовательского CSS."""
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Инициализация приложения
st.set_page_config(page_title="Анализ криптовалют", layout="wide")
load_custom_css()

st.title("Инструмент анализа криптовалют")

# Боковая панель
pairs = st.sidebar.multiselect(
    "Выберите криптовалютные пары",
    options=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
    default=["BTCUSDT", "ETHUSDT"],
    help="Выберите криптовалютные пары для анализа. Например, BTCUSDT — пара биткоина к доллару США."
)

interval = st.sidebar.selectbox(
    "Интервал",
    options=["1d", "1h", "1m"],
    help="Выберите временной интервал для анализа. Например, '1d' означает одну свечу за день."
)

chart_type = st.sidebar.selectbox(
    "Тип графика",
    options=["Линейный", "Свечной", "Баровый"],
    help=(
        "Линейный: Показывает общую тенденцию изменения цен. "
        "Свечной: Отображает открытие, закрытие, максимум и минимум цен за период. "
        "Баровый: Аналог свечного, но с другим визуальным представлением."
    )
)

# Преобразование выбранного типа в формат для функции
chart_type_mapping = {
    "Линейный": "line",
    "Свечной": "candlestick",
    "Баровый": "bar"
}

selected_chart_type = chart_type_mapping[chart_type]

# Параметр выбора периода анализа
period = st.sidebar.slider(
    "Введите период в днях",
    min_value=1,
    max_value=365,
    value=30,
    help="Определяет, за сколько дней будут загружены данные для анализа. Например, 30 дней — данные за последний месяц."
)

# Ползунок для настройки SMA
sma_window = st.sidebar.slider(
    "Период SMA",
    min_value=5,
    max_value=50,
    value=14,
    step=1,
    help="SMA (Simple Moving Average) — скользящее среднее. Чем больше период, тем больше сглаживается график."
)

# Ползунок для настройки EMA
ema_window = st.sidebar.slider(
    "Период EMA",
    min_value=5,
    max_value=50,
    value=14,
    step=1,
    help="EMA (Exponential Moving Average) — экспоненциальное скользящее среднее. Более короткий период делает EMA более чувствительным к последним изменениям цен."
)

# Преобразование типа графика в значения для функции
chart_type_mapping = {
    "Линейный": "line",
    "Свечной": "candlestick",
    "Баровый": "bar"
}
selected_chart_type = chart_type_mapping[chart_type]

# Загрузка данных
data_dict = {}
if st.sidebar.button("Загрузить данные"):
    for pair in pairs:
        with st.spinner(f"Загрузка данных для {pair}..."):
            data = get_historical_data(pair, interval="1d", start_str=f"{period} days ago UTC")
            if not data.empty:
                data_dict[pair] = data
                st.success(f"Данные для {pair} успешно загружены!")
            else:
                st.error(f"Ошибка загрузки данных для {pair}.")

# Вкладки интерфейса
tab1, tab2, tab3 = st.tabs(["Данные", "Графики", "Рекомендации"])

# Вкладка "Данные"
with tab1:
    st.header("Данные активов")
    for pair, data in data_dict.items():
        display_table(data, title=f"Данные для {pair}")

# Вкладка "Графики"
with tab2:
    st.header("Графики с индикаторами и объёмами")
    if data_dict:
        for pair, data in data_dict.items():
            st.subheader(f"График для {pair}")

            # Рассчитываем индикаторы
            data = calculate_moving_averages(data, sma_window)
            data = calculate_moving_averages(data, ema_window)

            # Построение графика
            plot = plot_with_indicators(
                data, title=f"График для {pair}",
                sma_window=sma_window,
                ema_window=ema_window,
                chart_type=selected_chart_type  # Верхний график меняется в зависимости от выбора
            )
            st.plotly_chart(plot, use_container_width=True, config={"scrollZoom": True})

            # Описание
            st.markdown("""
            **Описание:**
            - Верхний график: отображает цены с выбранным типом отображения (линейный, свечной, баровый).
            - Нижний график: объемы всегда отображаются в виде баров.
            - **SMA (Simple Moving Average):** Скользящее среднее для определения тренда.
            - **EMA (Exponential Moving Average):** Экспоненциальное скользящее среднее для более точного анализа.
            """)
    else:
        st.warning("Данные не загружены. Выберите активы и нажмите 'Загрузить данные'.")

# Вкладка "Рекомендации"
with tab3:
    st.header("Рекомендации")
    if data_dict:
        recommendations = generate_advanced_recommendations(data_dict)
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.warning("Данные не загружены. Выберите активы и нажмите 'Загрузить данные'.")
