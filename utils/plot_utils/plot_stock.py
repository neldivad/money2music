import plotly.graph_objects as go
import pandas as pd

def make_linechart(stock_data: pd.DataFrame):
    """Create a line chart for stock closing price."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines+markers',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title='Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_dark',
        showlegend=False
    )
    return fig

def make_barchart(stock_data: pd.DataFrame):
    """Create a bar chart for stock volume."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stock_data.index,
        y=stock_data['Volume'],
        name='Volume',
        marker_color='#ff7f0e',
        opacity=0.7
    ))
    fig.update_layout(
        title='Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_dark',
        showlegend=False
    )
    return fig 