from prophet import Prophet
import pandas as pd

def forecast_prices(data, periods=30):
    """
    Прогнозування цін з використанням бібліотеки Prophet.

    Параметри:
    data (DataFrame): Історичні дані активу з колонками "open_time" (час) і "close" (ціна).
    periods (int): Кількість періодів для прогнозування (у днях).

    Повертає:
    DataFrame: Прогнозні дані з колонками ["ds", "yhat", "yhat_lower", "yhat_upper"].
    """
    if data.empty or "open_time" not in data or "close" not in data:
        raise ValueError("Дані відсутні або не містять необхідних колонок 'open_time' і 'close'.")

    try:
        # Підготовка даних
        df = data[["open_time", "close"]].rename(columns={"open_time": "ds", "close": "y"})

        # Ініціалізація та навчання моделі Prophet
        model = Prophet()
        model.fit(df)

        # Створення майбутнього DataFrame для прогнозування
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as e:
        raise RuntimeError(f"Помилка під час прогнозування: {e}")

def evaluate_forecast(actual_data, forecast):
    """
    Оцінка якості прогнозу на основі історичних даних.

    Параметри:
    actual_data (DataFrame): Фактичні дані з колонками "open_time" і "close".
    forecast (DataFrame): Прогнозні дані з колонками "ds" і "yhat".

    Повертає:
    dict: Метрики оцінки прогнозу (MAE, MSE, RMSE).
    """
    if actual_data.empty or "open_time" not in actual_data or "close" not in actual_data:
        raise ValueError("Фактичні дані відсутні або не містять колонок 'open_time' і 'close'.")
    if forecast.empty or "ds" not in forecast or "yhat" not in forecast:
        raise ValueError("Прогнозні дані відсутні або не містять необхідних колонок.")

    try:
        # Співставлення прогнозів із фактичними даними
        merged_data = pd.merge(
            actual_data.rename(columns={"open_time": "ds", "close": "actual"}),
            forecast[["ds", "yhat"]],
            on="ds",
            how="inner"
        )

        # Розрахунок метрик
        merged_data["error"] = merged_data["actual"] - merged_data["yhat"]
        mae = merged_data["error"].abs().mean()  # Mean Absolute Error
        mse = (merged_data["error"] ** 2).mean()  # Mean Squared Error
        rmse = mse ** 0.5  # Root Mean Squared Error

        return {"MAE": mae, "MSE": mse, "RMSE": rmse}
    except Exception as e:
        raise RuntimeError(f"Помилка під час оцінки прогнозу: {e}")