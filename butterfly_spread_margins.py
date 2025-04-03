import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Acess Data

fichier = "/Users/quentinleriche/Desktop/Etudes/Skema/Structured Products/data.xlsx"

vol_tab = pd.read_excel(fichier, sheet_name="Volatility Surface")
rf_tab = pd.read_excel(fichier, sheet_name="Yield Curve")


# Set Black-Schole pricer

def BS_pricer(strike, vol, maturity, spot, rf_rate):

    d1 = (np.log(spot / strike) + (rf_rate + 0.5 * vol**2) * maturity) / (vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)
    
    call = spot * norm.cdf(d1) - strike * np.exp(-rf_rate * maturity) * norm.cdf(d2)
    put = strike * np.exp(-rf_rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    price = pd.DataFrame([{
        "Call": call,
        "Put": put,
    }])
    
    return price

def nearest_ten(a):
    return round(a / 10) * 10


# Calculate margins + PnLs plot

def margin_long_butterfly(long_call_1, short_call, long_call_2, maturity, capital_garenty, spot_underlying) :
        
    long_call_1 = nearest_ten(long_call_1)
    short_call = nearest_ten(short_call)
    long_call_2 = nearest_ten(long_call_2)
        
    vol = vol_tab.set_index("Strike").astype(float)
    rf_rate = float(rf_tab["Yield"].iloc[maturity-1])
    vol_long_call_2 = vol.loc[long_call_2, str(maturity)]
    vol_short_call = vol.loc[short_call, str(maturity)]
    vol_long_call_1 = vol.loc[long_call_1, str(maturity)]
        
        
    cost_garenty = capital_garenty / (1 + rf_rate)**(maturity) 
    
    cost_long_call_1 = BS_pricer(long_call_1, vol_long_call_1, maturity, spot_underlying, rf_rate)["Call"]
    cost_short_call = BS_pricer(short_call, vol_short_call, maturity, spot_underlying, rf_rate)["Call"]
    cost_long_call_2 = BS_pricer(long_call_2, vol_long_call_2, maturity, spot_underlying, rf_rate)["Call"]
    
    cost_of_options = cost_long_call_1 - 2 * cost_short_call + cost_long_call_2
        
        
    margin = 100 - cost_garenty - cost_of_options
        
    
       
    underlying_prices = np.linspace(50, 150, 500)
        
    pnl_1 = [max(price - long_call_1, 0) - cost_long_call_1 for price in underlying_prices]  
    pnl_2 = [2 * (cost_short_call - max(price - short_call, 0)) for price in underlying_prices]
    pnl_3 = [max(price - long_call_2, 0) - cost_long_call_2 for price in underlying_prices] 
        
    total_pnls = [K1_pnl + K2_pnl + K3_pnl for K1_pnl, K2_pnl, K3_pnl in zip(pnl_1, pnl_2, pnl_3)]

        
    plt.plot(underlying_prices, total_pnls, label='PnL', color='g', linewidth=2)
    plt.axhline(0, color='r', linestyle='--', linewidth=1)  # Baseline
    plt.title('Long Butterfly')
    plt.xlabel('Underlying Price')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True)
    
        
    return margin, plt


def margin_short_butterfly(short_call_1, long_call, short_call_2, maturity, capital_garenty, spot_underlying):
    
    short_call_1 = nearest_ten(short_call_1)
    long_call = nearest_ten(long_call)
    short_call_2 = nearest_ten(short_call_2)

    vol = vol_tab.set_index("Strike").astype(float)
    rf_rate = float(rf_tab["Yield"].iloc[maturity - 1])
    vol_short_call_1 = vol.loc[short_call_1, str(maturity)]
    vol_long_call = vol.loc[long_call, str(maturity)]
    vol_short_call_2 = vol.loc[short_call_2, str(maturity)]
    

    cost_garenty = capital_garenty / (1 + rf_rate)

    cost_short_call_1 = BS_pricer(short_call_1, vol_short_call_1, maturity, spot_underlying, rf_rate)["Call"]
    cost_long_call = BS_pricer(long_call, vol_long_call, maturity, spot_underlying, rf_rate)["Call"]
    cost_short_call_2 = BS_pricer(short_call_2, vol_short_call_2, maturity, spot_underlying, rf_rate)["Call"]

    cost_of_options =  2 * cost_long_call - cost_short_call_2 - cost_short_call_1


    margin = 100 - cost_garenty - cost_of_options



    underlying_prices = np.linspace(50, 150, 500)
    
    pnl_1 = [-max(price - short_call_1, 0) + cost_short_call_1 for price in underlying_prices]  
    pnl_2 = [2 * (max(price - long_call, 0) - cost_long_call) for price in underlying_prices]   
    pnl_3 = [-max(price - short_call_2, 0) + cost_short_call_2 for price in underlying_prices]

    total_pnls = [K1_pnl + K2_pnl + K3_pnl for K1_pnl, K2_pnl, K3_pnl in zip(pnl_1, pnl_2, pnl_3)]


    plt.plot(underlying_prices, total_pnls, label='PnL', color='b', linewidth=2)
    plt.axhline(0, color='r', linestyle='--', linewidth=1)  # Baseline
    plt.title('Short Butterfly')
    plt.xlabel('Underlying Price')
    plt.ylabel('PnL')
    plt.legend()
    plt.grid(True)

    return margin, plt

    
    
    
    
