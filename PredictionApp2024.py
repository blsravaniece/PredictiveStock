import dash
import dash_bootstrap_components as dbc
from dash import dash_table
from dash import dcc, html, Input, Output, State
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import random
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import requests
from textblob import TextBlob

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = 'Stock Market Dashboard'

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("Stock Price Dashboard", className="text-center my-4 text-primary"))),
        dbc.Row(dbc.Col(html.H6("Live Stock Ticker", className="mt-4"))),
        dbc.Row([
        dbc.Col(dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0))]),
        dbc.Row([
        dbc.Col(html.Div(id='live-ticker', style={'display': 'flex', 'overflowX': 'scroll'}))
                ]),
        dbc.Row(dbc.Col(html.H6("Enter Stock Ticker to get information", className="mt-4"))),        
        dbc.Row(
            [
                dbc.Col(dbc.Input(id='ticker-input', placeholder='Enter stock ticker', type='text'), width=4),
                dbc.Col(dbc.Button('Submit', id='submit-button', color='primary'), width=2)
            ], 
            className="mb-4"
        ),
        dbc.Row(
            [
                dbc.Col(html.Div(id='company-logo', style={'textAlign': 'center'}), width=2),
                dbc.Col(html.Div(id='company-info', className='text-white bg-dark p-2'), width=10)
            ]
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id='stock-price-graph', config={'displayModeBar': False}), className="my-4")
        ),
        dbc.Row(
            [
                dbc.Col(html.Div(id='news-section', className='text-white bg-dark p-2'), width=6),
                dbc.Col(html.Div(id='sentiment-section', className='text-white bg-dark p-2'), width=6)
            ]
        ),
        dbc.Row(dbc.Col(html.H6("Get Predicted Stock Price ", className="mt-4"))),
        dbc.Row(
            [
                dbc.Col(dcc.Input(id='predict-days', placeholder='Days to predict', type='number'), width=4),
                dbc.Col(dbc.Button('Predict', id='predict-button', color='primary'), width=2)
            ], 
            className="mb-4"
        ),
        dbc.Row(dbc.Col(dcc.Graph(id='prediction-graph', config={'displayModeBar': False}), className="my-4")),
        dbc.Row([
                  dbc.Col(html.Div(id='stock-prediction-table')),
                ]),
        
        dbc.Row([
        dbc.Col(html.H2("Previous Day's Equity Market"), className="mt-4")]),
        dbc.Row([
        dbc.Col(dash_table.DataTable(id='equity-market-table', columns=[], data=[], style_table={'overflowX': 'auto'}))
                ]),
    ],
    
    fluid=True,
    style={'backgroundColor': '#2C3E50'}
)

def fetch_news_and_sentiment(ticker):
    api_key = '44e532a29d39440a9e8c2b3698ad6fb1'
    url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data.get('articles', [])

    news_list = []
    sentiment_scores = []
    for article in articles[:5]:  # Limiting to top 5 news articles
        news_list.append(f"{article['title']} - {article['source']['name']}")
        analysis = TextBlob(article['description'] if article['description'] else '')
        sentiment_scores.append(analysis.sentiment.polarity)

    average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    sentiment_summary = 'Positive' if average_sentiment > 0 else 'Negative' if average_sentiment < 0 else 'Neutral'
    return news_list, sentiment_summary, average_sentiment

@app.callback(
    [
        Output('company-logo', 'children'),
        Output('company-info', 'children'),
        Output('stock-price-graph', 'figure'),
        Output('news-section', 'children'),
        Output('sentiment-section', 'children')
    ],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_company_info(n_clicks, ticker):
    if not ticker:
        return "", "", {}, "", ""
    stock = yf.Ticker(ticker)
    info = stock.info
    logo_url = info.get('logo_url', '')
    company_info = f"""
        **Company Name:** {info.get('longName', 'N/A')}  
        **Sector:** {info.get('sector', 'N/A')}  
        **Company Details:** {info.get('longBusinessSummary', 'N/A')}
        **Industry:** {info.get('industry', 'N/A')}
        **Website:** {info.get('website', 'N/A')}
    """
    
    df = stock.history(period='1y')
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                  open=df['Open'],
                  high=df['High'],
                  low=df['Low'],
                  close=df['Close'])])

    news_list, sentiment_summary, average_sentiment = fetch_news_and_sentiment(ticker)
    news_section = html.Ul([html.Li(news) for news in news_list])
    sentiment_section = f"Market Sentiment: **{sentiment_summary}** (Score: {average_sentiment:.2f})"
    
    return html.Img(src=logo_url, style={'height': '100px'}), dcc.Markdown(company_info), fig, news_section, dcc.Markdown(sentiment_section)

@app.callback(
    [Output('prediction-graph', 'figure'), 
     Output('stock-prediction-table', 'children')
    ],
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value'),
     dash.dependencies.State('predict-days', 'value')]
)
def predict_stock_prices(n_clicks, ticker, days):
    if not ticker or not days:
        return {}
    stock = yf.Ticker(ticker)
    df = stock.history(period='1y')
    
    # Prepare the data for LSTM
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Creating training data set
    train_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:train_data_len, :]
    
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    # Create the testing data set
    test_data = scaled_data[train_data_len - 60:, :]
    x_test = []
    y_test = data[train_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    
    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Predict future prices
    last_60_days = scaled_data[-60:]
    next_days_predictions = []
    news_list, sentiment_summary, average_sentiment = fetch_news_and_sentiment(ticker)
    sentiment_factor = 1 + average_sentiment  # Factor to adjust prediction based on sentiment
    for _ in range(days):
        next_input = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        next_pred = model.predict(next_input)
        next_pred = scaler.inverse_transform(next_pred) * sentiment_factor
        next_days_predictions.append(next_pred[0, 0])
        last_60_days = np.append(last_60_days[1:], next_pred, axis=0)
    
    # Plot the results
    prediction_df = pd.DataFrame({
        'Date': pd.date_range(start=df.index[-1], periods=days + 1, inclusive='right'),
        'Prediction': next_days_predictions
    })
    fig1 = px.line(prediction_df, x='Date', y='Prediction', title=f'{ticker} Stock Price Prediction')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Prediction'], mode='lines', name='Predicted Price'))
    table = dbc.Table.from_dataframe(prediction_df, striped=True, bordered=True, hover=True)
    return fig1, table
@app.callback(
    Output('live-ticker', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_ticker(n):
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA','SPCE','OXY','NIO','BABA','SOUN','SBUX','QCOM','SNOW','SMCI','DIS','NFLX']
    selected_symbols = random.sample(stock_symbols, 15)
    
    tickers = []
    for symbol in selected_symbols:
        try:
            df = yf.download(symbol, period='1d', interval='1d')
            if not df.empty:
                price = df['Close'][-1]
                tickers.append(html.Div(f'{symbol}: {price:.2f}', style={'margin': '0 10px'}))
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    return tickers

@app.callback(
    [Output('equity-market-table', 'columns'),
     Output('equity-market-table', 'data')],
    Input('interval-component', 'n_intervals')
)
def update_equity_market_table(n):
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB', 'NFLX', 'NVDA', 'ADBE', 'INTC']
    market_data = []

    for symbol in stock_symbols:
        try:
            df = yf.download(symbol, period='5d')
            if not df.empty:
                last_row = df.iloc[-2]  # Get the previous day's data
                market_data.append({
                    'Symbol': symbol,
                    'Last Price': last_row['Close'],
                    'Change': last_row['Close'] - last_row['Open'],
                    'Volume': last_row['Volume']
                })
        except Exception as e:
            market_data.append({
                'Symbol': symbol,
                'Last Price': 'Error',
                'Change': 'Error',
                'Volume': 'Error'
            })

    columns = [
        {'name': 'Symbol', 'id': 'Symbol'},
        {'name': 'Last Price', 'id': 'Last Price'},
        {'name': 'Change', 'id': 'Change'},
        {'name': 'Volume', 'id': 'Volume'}
    ]

    return columns, market_data



if __name__ == '__main__':
    app.run_server(debug=True)
