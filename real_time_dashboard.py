import websocket
import json
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Глобальное хранилище данных
global_data = pd.DataFrame()

def connect_websocket(pair, interval="1m"):
    """
    Подключение к WebSocket Binance.
    """
    try:
        url = f"wss://stream.binance.com:9443/ws/{pair.lower()}@kline_{interval}"
        ws = websocket.create_connection(url)
        return ws
    except Exception as e:
        raise ConnectionError(f"Ошибка подключения к WebSocket: {e}")

def fetch_real_time_data(ws):
    """
    Получение данных через WebSocket.
    """
    try:
        result = ws.recv()
        data = json.loads(result)
        kline = data['k']
        row = {
            "time": pd.to_datetime(kline["t"], unit="ms"),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }
        return row
    except Exception as e:
        raise RuntimeError(f"Ошибка получения данных: {e}")

def create_graph(data):
    """
    Построение графика.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["time"], y=data["close"], mode="lines", name="Цена закрытия"))
    fig.update_layout(
        template="plotly_white",
        title="График актива в реальном времени",
        xaxis_title="Время",
        yaxis_title="Цена (USD)"
    )
    return fig

# Интерфейс Dash
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("График в реальном времени"),
            html.Label("Выберите криптовалютную пару:"),
            dcc.Dropdown(
                id="symbol-selector",
                options=[
                    {"label": "BTC/USDT", "value": "BTCUSDT"},
                    {"label": "ETH/USDT", "value": "ETHUSDT"},
                    {"label": "BNB/USDT", "value": "BNBUSDT"}
                ],
                value="BTCUSDT",
                placeholder="Выберите пару"
            ),
            html.Label("Выберите интервал:"),
            dcc.Dropdown(
                id="interval-selector",
                options=[
                    {"label": "1 минута", "value": "1m"},
                    {"label": "5 минут", "value": "5m"},
                    {"label": "1 час", "value": "1h"}
                ],
                value="1m",
                placeholder="Выберите интервал"
            ),
            dcc.Graph(id="real-time-graph"),
            dcc.Interval(
                id="interval-update",
                interval=10*1000,  # Обновление каждые 10 секунд
                n_intervals=0
            )
        ], width=12)
    ])
])

@app.callback(
    Output("real-time-graph", "figure"),
    Input("symbol-selector", "value"),
    Input("interval-selector", "value"),
    Input("interval-update", "n_intervals")
)
def update_graph(pair, interval, n_intervals):
    """
    Обновление графика в реальном времени.
    """
    global global_data

    try:
        ws = connect_websocket(pair, interval)
        new_data = fetch_real_time_data(ws)
        ws.close()

        # Обновляем данные
        global_data = pd.concat([global_data, pd.DataFrame([new_data])]).tail(500)  # Храним только последние 500 точек
        return create_graph(global_data)

    except Exception as e:
        return go.Figure().update_layout(
            title=f"Ошибка загрузки данных: {e}",
            xaxis_title="Время",
            yaxis_title="Цена (USD)"
        )

if __name__ == "__main__":
    app.run_server(debug=True)
