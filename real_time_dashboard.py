import websocket
import json
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Глобальне сховище даних
global_data = pd.DataFrame()

def connect_websocket(pair, interval="1m"):
    """
    Підключення до WebSocket Binance.
    """
    try:
        url = f"wss://stream.binance.com:9443/ws/{pair.lower()}@kline_{interval}"
        ws = websocket.create_connection(url)
        return ws
    except Exception as e:
        raise ConnectionError(f"Помилка підключення до WebSocket: {e}")

def fetch_real_time_data(ws):
    """
    Отримання даних через WebSocket.
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
        raise RuntimeError(f"Помилка отримання даних: {e}")

def create_graph(data):
    """
    Побудова графіка.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["time"], y=data["close"], mode="lines", name="Ціна закриття"))
    fig.update_layout(
        template="plotly_white",
        title="Графік активу в реальному часі",
        xaxis_title="Час",
        yaxis_title="Ціна (USD)"
    )
    return fig

# Інтерфейс Dash
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("Графік у реальному часі"),
            html.Label("Оберіть криптовалютну пару:"),
            dcc.Dropdown(
                id="symbol-selector",
                options=[
                    {"label": "BTC/USDT", "value": "BTCUSDT"},
                    {"label": "ETH/USDT", "value": "ETHUSDT"},
                    {"label": "BNB/USDT", "value": "BNBUSDT"}
                ],
                value="BTCUSDT",
                placeholder="Оберіть пару"
            ),
            html.Label("Оберіть інтервал:"),
            dcc.Dropdown(
                id="interval-selector",
                options=[
                    {"label": "1 хвилина", "value": "1m"},
                    {"label": "5 хвилин", "value": "5m"},
                    {"label": "1 година", "value": "1h"}
                ],
                value="1m",
                placeholder="Оберіть інтервал"
            ),
            dcc.Graph(id="real-time-graph"),
            dcc.Interval(
                id="interval-update",
                interval=10*1000,  # Оновлення кожні 10 секунд
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
    Оновлення графіка у реальному часі.
    """
    global global_data

    try:
        ws = connect_websocket(pair, interval)
        new_data = fetch_real_time_data(ws)
        ws.close()

        # Оновлюємо дані
        global_data = pd.concat([global_data, pd.DataFrame([new_data])]).tail(500)  # Зберігаємо лише останні 500 точок
        return create_graph(global_data)

    except Exception as e:
        return go.Figure().update_layout(
            title=f"Помилка завантаження даних: {e}",
            xaxis_title="Час",
            yaxis_title="Ціна (USD)"
        )

if __name__ == "__main__":
    app.run_server(debug=True)