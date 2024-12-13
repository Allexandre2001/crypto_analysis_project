import numpy as np
import pandas as pd
import streamlit as st
from pygments.lexers import go
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from data.api import get_historical_data
from data.calculations import (
    generate_advanced_recommendations, calculate_moving_averages,
    calculate_var, calculate_atr, calculate_bollinger_bands, calculate_correlation_matrix, monte_carlo_simulation,
    calculate_bayesian_probabilities, calculate_bayes_laplace, calculate_savage, calculate_hurwicz, calculate_returns,
    calculate_sharpe_ratio, generate_trend_recommendations, highlight_risk_zones, detect_trends, clean_data,
    process_indicators
)
from data.visualization import (
    plot_price_and_volume_optimized, plot_comparison, display_table,
    plot_bollinger_bands, plot_correlation_matrix, plot_monte_carlo, plot_long_short,
    plot_criteria_results, plot_bayesian_probabilities, plot_risk_zones, plot_trends
)



def normalize_criteria_results(criteria_results):
    """
    Нормализация результатов критериев для корректного отображения.

    :param criteria_results: Словарь с результатами критериев.
    :return: Нормализованный DataFrame.
    """
    scaler = MinMaxScaler()

    # Преобразование словаря в DataFrame для нормализации
    df = pd.DataFrame(criteria_results)

    # Применяем MinMaxScaler для нормализации
    normalized_values = scaler.fit_transform(df)

    # Создаем новый DataFrame с нормализованными значениями
    normalized_df = pd.DataFrame(normalized_values, columns=df.columns, index=df.index)

    return normalized_df

def load_custom_css():
    """Загрузка пользовательского CSS для улучшения визуального оформления."""
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Игнорируем отсутствие файла

# Инициализация приложения
st.set_page_config(page_title="Инструмент анализа криптовалют", layout="wide")
st.title("Инструмент анализа криптовалют")

# Боковая панель
pairs = st.sidebar.multiselect(
    "Выберите криптовалютные пары",
    options=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
    default=["BTCUSDT"],
    help="Выберите криптовалютные пары для анализа."
)

interval = st.sidebar.selectbox(
    "Интервал",
    options=["1d", "1h", "1m"],
    help="Выберите временной интервал для анализа."
)

chart_type = st.sidebar.selectbox(
    "Тип графика",
    options=["Линейный", "Свечной", "Баровый"],
    help="Выберите тип графика для отображения цен."
)

chart_type_mapping = {"Линейный": "line", "Свечной": "candlestick", "Баровый": "bar"}
selected_chart_type = chart_type_mapping[chart_type]

# Периоды анализа
long_term_period = st.sidebar.slider(
    "Период долгосрочного анализа (в днях)",
    min_value=30, max_value=365, value=180,
    help="Период для долгосрочного анализа данных."
)

short_term_period = st.sidebar.slider(
    "Период краткосрочного анализа (в днях)",
    min_value=1, max_value=30, value=7,
    help="Период для краткосрочного анализа данных."
)

sma_window = st.sidebar.slider("Период SMA", 5, 50, 14, 1)
ema_window = st.sidebar.slider("Период EMA", 5, 50, 14, 1)

# Глобальное хранилище данных
data_dict = {}

def load_data():
    """Загрузка данных для выбранных криптовалютных пар."""
    for pair in pairs:
        with st.spinner(f"Загрузка данных для {pair}..."):
            data = get_historical_data(pair, interval=interval, start_str=f"{long_term_period} days ago UTC")
            if not data.empty:
                data = calculate_moving_averages(data, sma_window)
                data = calculate_moving_averages(data, ema_window)
                data_dict[pair] = data
                st.success(f"Данные для {pair} успешно загружены!")
            else:
                st.error(f"Ошибка загрузки данных для {pair}.")

if st.sidebar.button("Загрузить данные"):
    load_data()

# Вкладки интерфейса
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Данные", "Графики", "Индикаторы", "Сравнительный анализ", "Корреляционная матрица","Моделирование", "Байесовское моделирование","Риск и доходность", "Рекомендации"
])

# Вкладка "Данные"
with tab1:
    st.header("Данные активов")
    if data_dict:
        for pair, data in data_dict.items():
            # Очистка данных перед отображением
            cleaned_data = clean_data(data)
            display_table(cleaned_data, title=f"Данные для {pair}")
    else:
        st.warning("Данные не загружены. Выберите активы и нажмите 'Загрузить данные'.")

# Вкладка "Графики"
with tab2:
    st.header("Графики цен и объемов")
    if data_dict:
        for pair, data in data_dict.items():
            st.subheader(f"Графики для {pair}")

            # Генерируем единый график для синхронизации
            fig = plot_price_and_volume_optimized(
                data, interval=interval,
                sma_window=sma_window, ema_window=ema_window,
                chart_type=selected_chart_type
            )
            # Отображаем график
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

            # Описание
            st.markdown(f"""
                       **Описание для {pair}:**
                       - Верхний график: отображает цены с выбранным типом отображения (линейный, свечной, баровый).
                       - Нижний график: объемы отображаются отдельно.
                       - **SMA (Simple Moving Average):** Скользящее среднее для определения тренда.
                       - **EMA (Exponential Moving Average):** Экспоненциальное скользящее среднее для более точного анализа.
                       """)

            # Рассчитываем волатильность
            data["volatility"] = data["close"].pct_change().rolling(window=14).std()

            # Риск-зоны
            st.subheader(f"Зоны риска для {pair}")
            fig_risk = highlight_risk_zones(data, title=f"Зоны риска для {pair}")
            st.plotly_chart(fig_risk, use_container_width=True)

            # Анализ трендов
            data = detect_trends(data)
            st.subheader(f"Анализ трендов для {pair}")
            fig_trends = plot_trends(data, title=f"Анализ трендов для {pair}")
            st.plotly_chart(fig_trends, use_container_width=True)

            # Рекомендации
            st.write("**Рекомендации:**")
            if data["trend"].iloc[-1] == "up":
                st.write(f"Для {pair}: восходящий тренд. Рекомендуется держать или увеличить инвестиции.")
            elif data["trend"].iloc[-1] == "down":
                st.write(f"Для {pair}: нисходящий тренд. Рекомендуется уменьшить долю или зафиксировать прибыль.")
            else:
                st.write(f"Для {pair}: стабильный тренд. Нет необходимости в изменении позиции.")
    else:
        st.warning("Данные не загружены. Пожалуйста, выберите активы и нажмите 'Загрузить данные'.")

# Вкладка "Индикаторы"
with tab3:
    st.header("Индикаторы риска и волатильности")
    if data_dict:
        for pair, data in data_dict.items():
            try:
                st.subheader(f"Индикаторы для {pair}")

                # Расчет индикаторов
                indicators = process_indicators(data_dict, pair)

                # Вывод значений индикаторов
                st.write(f"Средний ATR (14 дней): {indicators['ATR_mean']:.2f}")
                st.write(f"Value at Risk (95%): {indicators['VaR_95']:.2%}")
                st.write(f"Sharpe Ratio: {indicators['Sharpe_Ratio']:.2f}")

                # Визуализация зон риска
                fig_risk_zones = plot_risk_zones(data_dict[pair], atr_threshold=0.02)
                st.plotly_chart(fig_risk_zones, use_container_width=True)

                # Визуализация полос Боллинджера
                fig_bollinger = plot_bollinger_bands(data_dict[pair], title=f"Bollinger Bands для {pair}")
                st.plotly_chart(fig_bollinger, use_container_width=True)

            except ValueError as e:
                st.warning(f"Ошибка при расчете индикаторов для {pair}: {e}")


# Вкладка "Сравнительный анализ"
with tab4:
    st.header("Сравнительный анализ долгосрочных и краткосрочных трендов")
    if data_dict:
        for pair in pairs:
            if pair in data_dict:
                # Загрузка данных для долгосрочного и краткосрочного анализа
                data_long = get_historical_data(pair, interval="1d", start_str=f"{long_term_period} days ago UTC")
                data_short = get_historical_data(pair, interval="1h", start_str=f"{short_term_period} days ago UTC")

                if data_long.empty or data_short.empty:
                    st.warning(f"Недостаточно данных для {pair}. Пропуск анализа.")
                    continue

                # Построение графиков
                st.subheader(f"Сравнительный анализ для {pair}")
                fig_comparison = plot_comparison(data_long, data_short, title=f"Сравнительный анализ для {pair}")
                st.plotly_chart(fig_comparison, use_container_width=True)

                # Волатильность и тренды
                long_volatility = data_long["close"].std()
                short_volatility = data_short["close"].std()

                long_mean = data_long["close"].mean()
                short_mean = data_short["close"].mean()

                # Текстовая интерпретация
                st.markdown(f"""
                **Результаты анализа для {pair}:**
                - Средняя цена за долгосрочный период: **${long_mean:.2f}**
                - Средняя цена за краткосрочный период: **${short_mean:.2f}**
                - Волатильность за долгосрочный период: **{long_volatility:.2f}**
                - Волатильность за краткосрочный период: **{short_volatility:.2f}**
                """)
                st.markdown("""
                **Дополнительно:**
                - Более высокая волатильность краткосрочных данных может указывать на спекулятивные движения.
                - Если долгосрочный тренд устойчивый, это говорит о стабильности актива.
                """)

                # Добавление дополнительных визуализаций
                fig_long_short = plot_long_short(data_long, data_short, pair)
                st.plotly_chart(fig_long_short, use_container_width=True)
    else:
        st.warning("Данные не загружены. Выберите активы и нажмите 'Загрузить данные'.")


# Вкладка "Корреляционная матрица"
with tab5:
    st.header("Корреляционная матрица")
    if len(data_dict) > 1:
        try:
            # Рассчитываем корреляционную матрицу
            correlation_matrix = calculate_correlation_matrix(data_dict)

            # Отображаем матрицу
            fig = plot_correlation_matrix(correlation_matrix, title="Корреляционная матрица активов")
            st.plotly_chart(fig, use_container_width=True)

            # Отображаем сырые данные в виде таблицы
            st.subheader("Числовое представление корреляции")
            st.dataframe(correlation_matrix)
        except ValueError as e:
            st.warning(f"Ошибка: {str(e)}")
    else:
        st.warning("Для расчета корреляционной матрицы выберите минимум два актива.")


# Вкладка "Моделирование"
with tab6:
    st.header("Моделирование методом Монте-Карло")
    if data_dict:
        for pair, data in data_dict.items():
            st.subheader(f"Моделирование для {pair}")

            # Параметры моделирования
            num_simulations = st.number_input(
                f"Количество сценариев для {pair}",
                min_value=100, max_value=5000, value=1000, step=100
            )
            num_days = st.slider(
                f"Количество дней для прогноза {pair}",
                min_value=10, max_value=365, value=30
            )

            # Выполняем моделирование
            simulated_prices = monte_carlo_simulation(data, num_simulations=num_simulations, num_days=num_days)

            # Визуализируем результаты
            fig = plot_monte_carlo(simulated_prices, title=f"Моделирование для {pair}")
            st.plotly_chart(fig, use_container_width=True)

            # Статистика
            st.write(f"Последняя цена: {data['close'].iloc[-1]:.2f} USD")
            st.write(f"Средняя прогнозируемая цена через {num_days} дней: {simulated_prices.iloc[-1].mean():.2f} USD")
            st.write(f"Минимальная прогнозируемая цена: {simulated_prices.iloc[-1].min():.2f} USD")
            st.write(f"Максимальная прогнозируемая цена: {simulated_prices.iloc[-1].max():.2f} USD")
    else:
        st.warning("Данные не загружены. Выберите активы и нажмите 'Загрузить данные'.")

# Вкладка "Байесовское моделирование"
with tab7:
    st.header("Байесовское обновление вероятностей")
    if data_dict:
        for pair, data in data_dict.items():
            st.subheader(f"Байесовский анализ для {pair}")

            # Рассчитываем вероятности
            probabilities = calculate_bayesian_probabilities(data)

            # Отображаем вероятности
            st.write(f"Вероятность повышения цены: {probabilities['up']:.2%}")
            st.write(f"Вероятность снижения цены: {probabilities['down']:.2%}")

            # Строим график
            fig = plot_bayesian_probabilities(probabilities, title=f"Вероятности для {pair}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Данные не загружены. Выберите активы и нажмите 'Загрузить данные'.")

with tab8:  # "Риск и доходность"
    st.header("Риск и доходность")
    if data_dict:
        for pair, data in data_dict.items():
            st.subheader(f"Оценка риска и доходности для {pair}")

            if "daily_return" not in data:
                data = calculate_returns(data)

            if data["daily_return"].dropna().empty:
                st.warning(f"Недостаточно данных для расчета риска и доходности для {pair}.")
                continue

            payout_matrix = pd.DataFrame(
                {
                    f"Сценарий {i+1}": data["daily_return"].dropna().tail(10).sample(10, replace=True).values
                    for i in range(10)
                },
                index=[f"Альтернатива {i+1}" for i in range(10)]
            )

            probabilities = np.random.dirichlet(np.ones(10), size=1)[0]

            bayes_results = calculate_bayes_laplace(payout_matrix, probabilities)
            savage_results = calculate_savage(payout_matrix)
            hurwicz_results = calculate_hurwicz(payout_matrix, alpha=0.5)

            criteria_results = {
                "Байес-Лаплас": bayes_results,
                "Сэвидж": savage_results,
                "Гурвиц (α=0.5)": hurwicz_results
            }

            # Нормализация критериев
            criteria_results_normalized = normalize_criteria_results(criteria_results)

            fig = plot_criteria_results(criteria_results_normalized, title=f"Риск и доходность для {pair}")
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

            st.markdown("""
            **Описание:**
            - **Байес-Лаплас**: оценивает средние выплаты по вероятностям.
            - **Сэвидж**: минимизирует максимальные потери.
            - **Гурвиц**: баланс между оптимизмом и пессимизмом.
            """)
    else:
        st.warning("Данные для оценки не загружены. Выберите активы и нажмите 'Загрузить данные'.")


with tab9:  # "Рекомендации"
    st.header("Рекомендации")
    if data_dict:
        for pair, data in data_dict.items():
            recommendations = generate_trend_recommendations(data)
            st.subheader(f"Рекомендации для {pair}")
            for rec in recommendations:
                st.markdown(f"- {rec}")
    else:
        st.warning("Данные не загружены. Выберите активы и нажмите 'Загрузить данные'.")