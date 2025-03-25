import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def main():
    st.title("Interactive Brownian Motion with Real Stock Data")

    st.sidebar.header("Simulation Controls")
    # Ticker selection
    all_tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "IBM", "ORCL"]
    tickers = st.sidebar.multiselect("Choose Tickers:", all_tickers, default=["MSFT", "AAPL"])

    # Number of historical days
    days = st.sidebar.number_input("Historical Days", min_value=10, max_value=365, value=60)
    # Number of simulation paths
    paths = st.sidebar.number_input("Number of Simulation Paths (P)", min_value=1, max_value=10000, value=100)
    # Number of timesteps
    timesteps = st.sidebar.number_input("Timesteps (N)", min_value=10, max_value=20000, value=1000)
    # Time horizon in months
    time_horizon_months = st.sidebar.number_input("Time Horizon (Months)", min_value=1, max_value=24, value=1)

    # Button to run simulation
    if st.sidebar.button("Generate Plots"):
        # For each selected ticker, do the pipeline
        for ticker in tickers:
            st.subheader(f"**{ticker}**")

            # 1. Fetch data from yfinance
            df = fetch_data(ticker, days)
            if df is None or df.empty:
                st.warning(f"No data returned for {ticker}.")
                continue

            # 2. Compute historical returns
            historical_returns = compute_returns(df)
            if len(historical_returns) < 2:
                st.warning(f"Not enough data points to compute returns for {ticker}.")
                continue

            # 3. Simulate future returns using Brownian Motion
            sim_returns = simulate_returns(historical_returns, df["Adj Close"].iloc[-1],
                                           paths, timesteps, time_horizon_months)

            # 4. Plot the histograms side by side
            fig = create_histogram_figure(historical_returns, sim_returns)
            st.plotly_chart(fig, use_container_width=True)

def fetch_data(ticker, days):
    """
    Fetches ~'days' of historical data for 'ticker' using yfinance.
    """
    try:
        # end date: today, start date: 'days' days ago
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=days)

        # Download data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df is None or df.empty:
            return None
        
        # Ensure we have 'Adj Close'
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def compute_returns(df):
    """
    Computes daily returns (Adj Close(t) / Adj Close(t-1) - 1) from a DataFrame
    with an 'Adj Close' column.
    """
    prices = df["Adj Close"].values
    returns = (prices[1:] / prices[:-1]) - 1
    return returns

def simulate_returns(historical_returns, last_price, paths, timesteps, months):
    """
    Performs a simple Brownian Motion simulation of final returns
    based on historical mean and std.
    """
    # Mean and standard deviation of historical daily returns
    hist_mean = np.mean(historical_returns)
    hist_std = np.std(historical_returns, ddof=1)

    # Convert months to fraction of a year
    T = months / 12.0
    dt = T / timesteps

    simulated_final_prices = []
    for _ in range(paths):
        price = last_price
        for _ in range(timesteps):
            dW = np.random.normal(0, np.sqrt(dt))
            # Simple "Euler-Maruyama" style update
            price += price * (hist_mean * dt + hist_std * dW)
        simulated_final_prices.append(price)

    # Convert final prices to returns relative to last historical price
    sim_returns = [p / last_price - 1 for p in simulated_final_prices]
    return sim_returns

def create_histogram_figure(historical_returns, simulated_returns):
    """
    Creates a Plotly figure with two overlapping histograms:
    - Historical Returns
    - Simulated (Predicted) Returns
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=historical_returns,
        name="Historical Returns",
        marker_color="red",
        opacity=0.5
    ))

    fig.add_trace(go.Histogram(
        x=simulated_returns,
        name="Predicted Returns",
        marker_color="limegreen",
        opacity=0.5
    ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Return",
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=0.7, y=0.95)
    )

    return fig

if __name__ == "__main__":
    main()
