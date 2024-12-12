from prophet import Prophet
import pandas as pd

def forecast_prices(data, periods=30):
    """
    Прогнозирование цен с использованием библиотеки Prophet.

    Параметры:
    data (DataFrame): Исторические данные актива с колонками "open_time" (время) и "close" (цена).
    periods (int): Количество периодов для прогнозирования (в днях).

    Возвращает:
    DataFrame: Прогнозные данные с колонками ["ds", "yhat", "yhat_lower", "yhat_upper"].
    """
    if data.empty or "open_time" not in data or "close" not in data:
        raise ValueError("Данные отсутствуют или не содержат необходимые колонки 'open_time' и 'close'.")

    try:
        # Подготовка данных
        df = data[["open_time", "close"]].rename(columns={"open_time": "ds", "close": "y"})

        # Инициализация и обучение модели Prophet
        model = Prophet()
        model.fit(df)

        # Создание будущего DataFrame для прогнозирования
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as e:
        raise RuntimeError(f"Ошибка при прогнозировании: {e}")

def evaluate_forecast(actual_data, forecast):
    """
    Оценка качества прогноза на основе исторических данных.

    Параметры:
    actual_data (DataFrame): Фактические данные с колонками "open_time" и "close".
    forecast (DataFrame): Прогнозные данные с колонками "ds" и "yhat".

    Возвращает:
    dict: Метрики оценки прогноза (MAE, MSE, RMSE).
    """
    if actual_data.empty or "open_time" not in actual_data or "close" not in actual_data:
        raise ValueError("Фактические данные отсутствуют или не содержат колонку 'open_time' и 'close'.")
    if forecast.empty or "ds" not in forecast or "yhat" not in forecast:
        raise ValueError("Прогнозные данные отсутствуют или не содержат необходимые колонки.")

    try:
        # Сопоставление прогнозов с фактическими данными
        merged_data = pd.merge(
            actual_data.rename(columns={"open_time": "ds", "close": "actual"}),
            forecast[["ds", "yhat"]],
            on="ds",
            how="inner"
        )

        # Расчёт метрик
        merged_data["error"] = merged_data["actual"] - merged_data["yhat"]
        mae = merged_data["error"].abs().mean()  # Mean Absolute Error
        mse = (merged_data["error"] ** 2).mean()  # Mean Squared Error
        rmse = mse ** 0.5  # Root Mean Squared Error

        return {"MAE": mae, "MSE": mse, "RMSE": rmse}
    except Exception as e:
        raise RuntimeError(f"Ошибка при оценке прогноза: {e}")
