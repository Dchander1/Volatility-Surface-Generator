import streamlit as st
import numpy as np  
import plotly.graph_objects as go
import yfinance as yf 
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility
from datetime import datetime, timedelta
from scipy.interpolate import griddata

st.title("Implied Volatility Surface Generator")

# Sidebar inputs
with st.sidebar: 
    ticker = st.text_input("Enter Ticker Symbol", value="NFLX").upper()
    stock = yf.Ticker(ticker)
    stock_history = stock.history(period="1d")

    if stock_history.empty:
        st.error(f"No historical data found for {ticker}. Please check the ticker symbol.")
        st.stop()  # Stops execution if no stock data is found

    stock_price = stock_history['Close'].iloc[-1]  # Get last closing price
    today = datetime.today()

    # Fetch available expiration dates
    try:
        expiration_dates = [
            date for date in stock.options 
            if today + timedelta(days=7) <= datetime.strptime(date, "%Y-%m-%d") <= today + timedelta(days=730)]
    except:
        expiration_dates = []

    if not expiration_dates:
        st.error("No available options expiring after a week.")
        st.stop()

    risk_free_rate = st.number_input("Enter risk-free rate (e.g. 0.05 for 5%)", value=0.05, min_value=0.0)

    min_input = st.number_input("Minimum Strike Price (as % of Spot Price)", value=0.8, min_value=0.5, max_value=1.9)  
    max_input = st.number_input("Maximum Strike Price (as % of Spot Price)", value=1.2, min_value=0.6, max_value=2.0)

    lower_bound = min_input * stock_price
    upper_bound = max_input * stock_price

    if lower_bound < 0.5 * stock_price or upper_bound > 2.0 * stock_price:
        st.error("Strike price must be between 50% and 200% of the spot price.")
        st.stop()

# Function to get expiration dates and time to maturity
def fetch_option_data(ticker, risk_free_rate):
    all_data = []

    for exp_date in expiration_dates:
        expiration_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
        time_to_maturity_years = (expiration_datetime - today).days / 365.0  
        time_to_maturity_weeks = (expiration_datetime - today).days / 7.0  

        try:
            option_chain = stock.option_chain(exp_date)
            calls = option_chain.calls[['strike', 'ask']].copy()
            puts = option_chain.puts[['strike', 'ask']].copy()
        except:
            continue  # Skip if options data is unavailable

        calls['time_to_maturity'] = time_to_maturity_years
        puts['time_to_maturity'] = time_to_maturity_years
        calls['time_to_maturity_weeks'] = time_to_maturity_weeks
        puts['time_to_maturity_weeks'] = time_to_maturity_weeks
        calls['option_type'] = 'c'
        puts['option_type'] = 'p'

        # Filter strikes within the user-defined range
        calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)]
        puts = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound)]

        all_data.append(calls)
        all_data.append(puts)

    if all_data:
        option_data = pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=['strike', 'ask', 'time_to_maturity', 'option_type', 'implied_volatility'])

    # Function to compute implied volatility
    def compute_iv(row):    
        try:
            return implied_volatility(row['ask'], stock_price, row['strike'], row['time_to_maturity'], risk_free_rate, row['option_type'])
        except:
            return np.nan

    if not option_data.empty:
        option_data['implied_volatility'] = option_data.apply(compute_iv, axis=1)

    return option_data 

# Function to plot IV surface
def plot_iv_surface(option_data):
    option_data = option_data.dropna(subset=['implied_volatility'])  # Remove NaNs
    if option_data.empty:
        st.write("No valid options data available, displaying placeholder graph.")
        strikes = np.linspace(stock_price * 0.8, stock_price * 1.2, 10)
        times = np.linspace(1, 52, 10)
        T_grid, S_grid = np.meshgrid(times, strikes)
        IV_grid = np.full_like(T_grid, 0.5)
    else:
        strikes = option_data['strike'].values
        times = option_data['time_to_maturity_weeks'].values
        implied_vols = option_data['implied_volatility'].values

        # Create mesh grid
        T_grid, S_grid = np.meshgrid(np.linspace(times.min(), times.max(), 50), 
                                     np.linspace(strikes.min(), strikes.max(), 50))

        try:
            IV_grid = griddata((times, strikes), implied_vols, (T_grid, S_grid), method='cubic')
            if np.isnan(IV_grid).all():  # If cubic fails, use linear
                IV_grid = griddata((times, strikes), implied_vols, (T_grid, S_grid), method='linear')
        except:
            IV_grid = griddata((times, strikes), implied_vols, (T_grid, S_grid), method='nearest')

    fig = go.Figure(data=[go.Surface(z=IV_grid, x=T_grid, y=S_grid, colorscale='Viridis')])
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Time to Expiration (Weeks)",
            yaxis_title="Strike Price",
            zaxis_title="Implied Volatility"
        ),
        margin=dict(l=50, r=50, b=50, t=50),
        height=800, width=1200
    )
    st.plotly_chart(fig, use_container_width=True)

# Main execution
if ticker:
    option_data = fetch_option_data(ticker, risk_free_rate)
    if not option_data.empty:
        plot_iv_surface(option_data)
    else:
        st.write("No valid options data available for plotting.")