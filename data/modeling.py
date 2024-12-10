from prophet import Prophet
import pandas as pd

def forecast_prices(data, periods=30):
    """
    Прогнозирование цен с использованием библиотеки Prophet.
    """
    df = data[["open_time", "close"]].rename(columns={"open_time": "ds", "close": "y"})
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast
