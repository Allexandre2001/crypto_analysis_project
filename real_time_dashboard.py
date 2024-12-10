import websocket
import json
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Хранилище данных
global_data = pd.DataFrame()


def connect_websocket(pair):
    """Подключение к Binance WebSocket."""
    url = f"wss://stream.binance.com:9443/ws/{pair.lower()}@kline_1m"
    ws = websocket.create_connection(url)
    return ws


def fetch_real_time_data(ws):
    """Получение данных через WebSocket."""
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


# Построение графика
def create_graph(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["time"], y=data["close"], mode="lines", name="Цена закрытия"))
    fig.update_layout(template="plotly_white", title="График актива в реальном времени")
    return fig


app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("График в реальном времени"),
            dcc.Dropdown(
                id="symbol-selector",
                options=[
                    {"label": "BTC/USDT", "value": "BTCUSDT"},
                    {"label": "ETH/USDT", "value": "ETHUSDT"}
                ],
                value="BTCUSDT",
                placeholder="Выберите пару"
            ),
            dcc.Graph(id="real-time-graph"),
        ])
    ])
])


@app.callback(
    Output("real-time-graph", "figure"),
    Input("symbol-selector", "value")
)
def update_graph(pair):
    ws = connect_websocket(pair)
    global global_data
    new_data = fetch_real_time_data(ws)
    ws.close()

    # Обновляем данные
    global_data = pd.concat([global_data, pd.DataFrame([new_data])]).tail(100)
    return create_graph(global_data)


if __name__ == "__main__":
    app.run_server(debug=True)
