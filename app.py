import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from data.api import get_historical_data
from data.calculations import (
    calculate_moving_averages,
    calculate_correlation_matrix, monte_carlo_simulation,
    calculate_bayesian_probabilities, generate_trend_recommendations, highlight_risk_zones, detect_trends, clean_data,
    process_indicators
)
from data.visualization import (
    plot_price_and_volume_optimized, plot_comparison, display_table,
    plot_bollinger_bands, plot_correlation_matrix, plot_monte_carlo, plot_long_short,
    plot_bayesian_probabilities, plot_risk_zones, plot_trends
)



def normalize_criteria_results(criteria_results):
    """
    Нормалізує значення критеріїв у діапазон від 0 до 1.

    Аргументи:
    - criteria_results: словник з назвами критеріїв та їх значеннями.

    Повертає:
    - Нормалізовані значення критеріїв.
    """
    scaler = MinMaxScaler()

    # Перетворення словника в DataFrame для нормалізації
    df = pd.DataFrame(criteria_results)

    # Застосовуємо MinMaxScaler для нормалізації
    normalized_values = scaler.fit_transform(df)

    # Створюємо новий DataFrame з нормалізованими значеннями
    normalized_df = pd.DataFrame(normalized_values, columns=df.columns, index=df.index)

    return normalized_df

def load_custom_css():
    """Завантаження власного CSS для покращення вигляду."""
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Ігноруємо відсутність файлу

# Ініціалізація програми
st.set_page_config(page_title="Інструмент аналізу криптовалют", layout="wide")
st.title("Інструмент аналізу криптовалют")

# Бокова панель
pairs = st.sidebar.multiselect(
    "Оберіть криптовалютні пари",
    options=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
    default=["BTCUSDT"],
    help="Оберіть криптовалютні пари для аналізу."
)

interval = st.sidebar.selectbox(
    "Інтервал",
    options=["1d", "1h", "1m"],
    help="Оберіть часовий інтервал для аналізу."
)

chart_type = st.sidebar.selectbox(
    "Тип графіка",
    options=["Лінійний", "Свічковий", "Баровий"],
    help="Оберіть тип графіка для відображення цін."
)

chart_type_mapping = {"Лінійний": "line", "Свічковий": "candlestick", "Баровий": "bar"}
selected_chart_type = chart_type_mapping[chart_type]

# Параметри аналізу
long_term_period = st.sidebar.slider(
    "Період довгострокового аналізу (в днях)",
    min_value=30, max_value=365, value=180,
    help="Період для довгострокового аналізу даних."
)

short_term_period = st.sidebar.slider(
    "Період короткострокового аналізу (в днях)",
    min_value=1, max_value=30, value=7,
    help="Період для короткострокового аналізу даних."
)

sma_window = st.sidebar.slider("Період SMA", 5, 50, 14, 1)
ema_window = st.sidebar.slider("Період EMA", 5, 50, 14, 1)

# Глобальне сховище даних
data_dict = {}

def load_data():
    """Завантаження даних для обраних криптовалютних пар."""
    for pair in pairs:
        with st.spinner(f"Завантаження даних для {pair}..."):
            data = get_historical_data(pair, interval=interval, start_str=f"{long_term_period} days ago UTC")
            if not data.empty:
                data = calculate_moving_averages(data, sma_window)
                data = calculate_moving_averages(data, ema_window)
                data_dict[pair] = data
                st.success(f"Дані для {pair} успішно завантажено!")
            else:
                st.error(f"Помилка завантаження даних для {pair}.")

if st.sidebar.button("Завантажити дані"):
    load_data()


# Вкладки інтерфейсу
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,= st.tabs([
     "Дані", "Графіки", "Індикатори", "Порівняльний аналіз", "Кореляційна матриця",
    "Моделювання", "Байєсове моделювання", "Рекомендації"])

# Вкладка "Дані"
with tab1:
    st.header("Дані активів")
    if data_dict:
        for pair, data in data_dict.items():
            # Очищення даних перед відображенням
            cleaned_data = clean_data(data)
            display_table(cleaned_data, title=f"Дані для {pair}")
    else:
        st.warning("Дані не завантажені. Оберіть активи і натисніть 'Завантажити дані'.")

# Вкладка "Графіки"
with tab2:
    st.header("Графіки цін і обсягів")
    if data_dict:
        for pair, data in data_dict.items():
            st.subheader(f"Графіки для {pair}")

            # Генеруємо графік для синхронізації
            fig = plot_price_and_volume_optimized(
                data, interval=interval,
                sma_window=sma_window, ema_window=ema_window,
                chart_type=selected_chart_type
            )
            # Відображаємо графік
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

            # Опис
            st.markdown(f"""
                        **Опис для {pair}:**
                        - Верхній графік: відображає ціни з обраним типом графіка (лінійний, свічковий, стовпчиковий).
                        - Нижній графік: обсяги відображаються окремо.
                        - **SMA (Просте ковзне середнє):** Використовується для виявлення трендів.
                        - **EMA (Експоненційне ковзне середнє):** Для точнішого аналізу трендів.
                        """)

            # Розрахунок волатильності
            data["volatility"] = data["close"].pct_change().rolling(window=14).std()

            # Зони ризику
            st.subheader(f"Зони ризику для {pair}")
            fig_risk = highlight_risk_zones(data, title=f"Зони ризику для {pair}")
            st.plotly_chart(fig_risk, use_container_width=True)

            # Аналіз трендів
            data = detect_trends(data)
            st.subheader(f"Аналіз трендів для {pair}")
            fig_trends = plot_trends(data, title=f"Аналіз трендів для {pair}")
            st.plotly_chart(fig_trends, use_container_width=True)

            # Рекомендації
            st.write("**Рекомендації:**")
            if data["trend"].iloc[-1] == "up":
                st.write(f"Для {pair}: висхідний тренд. Рекомендується тримати або збільшити інвестиції.")
            elif data["trend"].iloc[-1] == "down":
                st.write(f"Для {pair}: низхідний тренд. Рекомендується зменшити долю або зафіксувати прибуток.")
            else:
                st.write(f"Для {pair}: стабільний тренд. Немає необхідності змінювати позицію.")
    else:
        st.warning("Дані не завантажені. Будь ласка, оберіть активи і натисніть 'Завантажити дані'.")

# Вкладка "Індикатори"
with tab3:
    st.header("Індикатори ризику і волатильності")
    if data_dict:
        for pair, data in data_dict.items():
            try:
                st.subheader(f"Індикатори для {pair}")

                # Розрахунок індикаторів
                indicators = process_indicators(data_dict, pair)

                # Виведення значень індикаторів
                st.write(f"Середній ATR (14 днів): {indicators['ATR_mean']:.2f}")
                st.write(f"Value at Risk (95%): {indicators['VaR_95']:.2%}")
                st.write(f"Sharpe Ratio: {indicators['Sharpe_Ratio']:.2f}")

                # Візуалізація зон ризику
                fig_risk_zones = plot_risk_zones(data_dict[pair], atr_threshold=0.02)
                st.plotly_chart(fig_risk_zones, use_container_width=True)

                # Візуалізація смуг Боллінджера
                fig_bollinger = plot_bollinger_bands(data_dict[pair], title=f"Bollinger Bands для {pair}")
                st.plotly_chart(fig_bollinger, use_container_width=True)

            except ValueError as e:
                st.warning(f"Помилка при розрахунку індикаторів для {pair}: {e}")


# Вкладка "Порівняльний аналіз"
with tab4:
    st.header("Порівняльний аналіз довгострокових та короткострокових трендів")
    if data_dict:
        for pair in pairs:
            if pair in data_dict:
                # Завантаження даних для довгострокового та короткострокового аналізу
                data_long = get_historical_data(pair, interval="1d", start_str=f"{long_term_period} days ago UTC")
                data_short = get_historical_data(pair, interval="1h", start_str=f"{short_term_period} days ago UTC")

                if data_long.empty or data_short.empty:
                    st.warning(f"Недостатньо даних для {pair}. Пропуск аналізу.")
                    continue

                # Побудова графіків
                st.subheader(f"Порівняльний аналіз для {pair}")
                fig_comparison = plot_comparison(data_long, data_short, title=f"Порівняльний аналіз для {pair}")
                st.plotly_chart(fig_comparison, use_container_width=True)

                # Волатильність та тренди
                long_volatility = data_long["close"].std()
                short_volatility = data_short["close"].std()

                long_mean = data_long["close"].mean()
                short_mean = data_short["close"].mean()

                # Текстова інтерпретація
                st.markdown(f"""
                **Результати аналізу для {pair}:**
                - Середня ціна за довгостроковий період: **${long_mean:.2f}**
                - Середня ціна за короткостроковий період: **${short_mean:.2f}**
                - Волатильність за довгостроковий період: **{long_volatility:.2f}**
                - Волатильність за короткостроковий період: **{short_volatility:.2f}**
                """)
                st.markdown("""
                **Додатково:**
                - Вища волатильність короткострокових даних може вказувати на спекулятивні рухи.
                - Якщо довгостроковий тренд є стійким, це говорить про стабільність активу.
                """)
                # Додавання додаткових візуалізацій
                fig_long_short = plot_long_short(data_long, data_short, pair)
                st.plotly_chart(fig_long_short, use_container_width=True)
    else:
        st.warning("Дані не завантажені. Оберіть активи та натисніть 'Завантажити дані'.")


# Вкладка "Кореляційна матриця"
with tab5:
    st.header("Кореляційна матриця")

    if len(data_dict) > 1:
        try:
            # Розраховуємо кореляційну матрицю
            correlation_matrix = calculate_correlation_matrix(data_dict)

            # Додаємо опис
            st.write("""
             **Опис:** Кореляційна матриця показує зв'язок між активами.
             Позитивне значення свідчить про пряму залежність, а негативне — зворотну.
             Значення варіюються від –1 (повна зворотна залежність) до 1 (повна пряма залежність).

             Наприклад:
             - Кореляція 0.7 та вище говорить про сильну залежність.
             - Значення від 0.3 до 0.7 показують помірну залежність.
             - Нижче 0.3 – слабка чи відсутня залежність.
             """)

            # Відображаємо матрицю
            st.subheader("Кореляційна матриця активів")
            fig = plot_correlation_matrix(correlation_matrix, title="Кореляційна матриця активів")
            st.plotly_chart(fig, use_container_width=True)

            # Відображаємо числові дані у вигляді таблиці
            st.subheader("Числове уявлення кореляції")
            st.dataframe(correlation_matrix.round(4))

            # Додаткове пояснення
            st.write("""
             **Що використовується для розрахунків:**
             - Історичні дані щодо цін обраних активів.
             - Перетворені дані для обчислення денних змін (прибутковостей).

             **Результати:**
             – Матриця наочно демонструє кореляцію між активами.
             - Висока кореляція може означати, що активи рухаються схожим чином, а низька чи негативна – про їхню незалежність.
             """)
        except ValueError as e:
            st.warning(f"Помилка: {str(e)}")
    else:
        st.warning("Для розрахунку кореляційної матриці виберіть мінімум два активи.")

# Вкладка "Моделювання"
with tab6:
    st.header("Моделювання методом Монте-Карло")
    if data_dict:
        for pair, data in data_dict.items():
            st.subheader(f"Моделювання для {pair}")

            # Параметри моделювання
            num_simulations = st.number_input(
                f"Кількість сценаріїв для {pair}",
                min_value=100, max_value=5000, value=1000, step=100
            )
            num_days = st.slider(
                f"Кількість днів для прогнозу {pair}",
                min_value=10, max_value=365, value=30
            )

            # Виконуємо моделювання
            simulated_prices = monte_carlo_simulation(data, num_simulations=num_simulations, num_days=num_days)

            # Візуалізуємо результати
            fig = plot_monte_carlo(simulated_prices, title=f"Моделювання для {pair}")
            st.plotly_chart(fig, use_container_width=True)

            # Статистика
            st.write(f"Остання ціна: {data['close'].iloc[-1]:.2f} USD")
            st.write(f"Середня прогнозована ціна через {num_days} днів: {simulated_prices.iloc[-1].mean():.2f} USD")
            st.write(f"Мінімальна прогнозована ціна: {simulated_prices.iloc[-1].min():.2f} USD")
            st.write(f"Максимальна прогнозована ціна: {simulated_prices.iloc[-1].max():.2f} USD")
    else:
        st.warning("Дані не завантажені. Виберіть активи та натисніть 'Завантажити дані'.")

# Вкладка "Байєсівське моделювання"
with tab7:
    st.header("Байєсівське оновлення ймовірностей")
    if data_dict:
        for pair, data in data_dict.items():
            st.subheader(f"Байєсівський аналіз для {pair}")

            # Розраховуємо ймовірності
            probabilities = calculate_bayesian_probabilities(data)

            # Відображаємо ймовірності
            st.write(f"Ймовірність підвищення ціни: {probabilities['up']:.2%}")
            st.write(f"Ймовірність зниження ціни: {probabilities['down']:.2%}")

            # Будуємо графік
            fig = plot_bayesian_probabilities(probabilities, title=f"Ймовірності для {pair}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Дані не завантажені. Виберіть активи та натисніть 'Завантажити дані'.")

# Вкладка "Рекомендації"
with tab8:
    st.header("Рекомендації")
    if data_dict:
        for pair, data in data_dict.items():
            recommendations = generate_trend_recommendations(data)
            st.subheader(f"Рекомендації для {pair}")
            for rec in recommendations:
                st.markdown(f"- {rec}")
    else:
        st.warning("Дані не завантажені. Виберіть активи та натисніть 'Завантажити дані'.")
