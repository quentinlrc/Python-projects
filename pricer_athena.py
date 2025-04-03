import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# path 0 - no cover
# path 10 - hit


# Parameters

initial_index_level = 100  
performance_trigger_level = 120  # (120% of initial index level)
capital_protection_level = 70  # (70% of initial index level)
coupon = 0.12  
num_simulations = 10000  
num_observations = 8  # (over 2 years)
risk_free_rate = 0.025  
volatility = 0.20 
dividend_yield = 0.04


#Simulation for index paths : geometric brownian motion
 
dt = 1 / 252  # Daily time step
time_steps = 252 * 2  # Two years of daily steps
index_paths = np.zeros((num_simulations, time_steps + 1))
index_paths[:, 0] = initial_index_level
observation_dates = np.arange(63, time_steps + 1, 63)

for i in range(1, time_steps + 1):
    z = np.random.standard_normal(num_simulations)
    index_paths[:, i] = index_paths[:, i - 1] * np.exp(
        (risk_free_rate - dividend_yield - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)


# Plot one specific path

def plot_path(idx):

    plt.figure(figsize=(10,6))
    
    plt.plot(index_paths[idx], label='Simulated Index Path')

    plt.axhline(y=performance_trigger_level, color='g', linestyle='--', label='Performance Trigger Level')
    plt.axhline(y=capital_protection_level, color='r', linestyle='--', label='Capital Protection Level')

    performance_touched = False
   
    for i, level in enumerate(index_paths[idx]):
        if i in observation_dates:  
            if level >= performance_trigger_level and not performance_touched:
                plt.axvline(x=i, color='g', linestyle='--', linewidth=3)
                plt.text(i + 5, performance_trigger_level - 30, 'PERFORMANCE TRIGGER HIT', color='g', fontsize=12, rotation=90, va='bottom')
                performance_touched = True
                
    if index_paths[idx, time_steps] < capital_protection_level and not performance_touched:
        plt.axvline(x=i, color='r', linestyle='--', linewidth=3)
        plt.text(480, capital_protection_level + 5, 'UNDER CAPITAL GARENTY', color='r', fontsize=12, rotation=90, va='bottom')


    for i, date in enumerate(observation_dates):
        plt.axvline(x=date, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Time (days)')
    plt.ylabel('Index Level')
    plt.title('Simulated Index Path')
    plt.legend()
    plt.grid(True)
    
    return plt


# Price using Monte-Carlo

payoff = np.zeros(num_simulations)
performance_trigger_hit = np.zeros(num_simulations, dtype=bool)

for j in range(num_simulations):
    
    for k in range(num_observations):
        
        if index_paths[j, observation_dates[k]] >= performance_trigger_level:
            payoff[j] = 100 + 100 * coupon
            performance_trigger_hit[j] = True
            break
    else:
        
        if index_paths[j, -1] >= capital_protection_level:
            payoff[j] = 100
        else:
            payoff[j] = 100 * (index_paths[j, -1] / initial_index_level)  


discounted_payoff = np.exp(-2 * risk_free_rate) * payoff

note_price = np.mean(discounted_payoff)

print(note_price)


# BONUS : payoff distribution

payoff_min = int(np.floor(payoff.min()))
payoff_max = int(np.ceil(payoff.max()))
num_bins = payoff_max - payoff_min + 1  

plt.hist(payoff, bins=num_bins, range=(payoff_min, payoff_max), alpha=0.75, color='blue', edgecolor='black')
plt.xlabel('Payoff')
plt.ylabel('Frequency (log)')
plt.title('Distribution of Payoffs')
plt.yscale("log")
plt.grid(True)

plt.axvline(x=note_price, color='red', linestyle='--', linewidth=2, label='Note Price: ' + f"{note_price:.2f}")

plt.legend()

plt.plot()



    
    



