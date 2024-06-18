import numpy as np
import time
from scipy.stats import skew, kurtosis
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
import random


# Set the random seed number
seed_value = 188487  # random.randint(0, 10000000)
random.seed(seed_value)
np.random.seed(seed_value)

# Print the current date and time
current_datetime = datetime.now()
print("Current date and time:", current_datetime)
start_time = time.time()


# Parameters
num_simulations = 10000
num_days = 100  # number of days
initial_price = 100  # stock price on day 0
volatility = 0.1  # magnitude of change in price per dollar of price
probability_up = 0.5  # probability of the price going up on any particular day

# Initialize storage for simulation results
all_prices = []
positive_log_counts = []
negative_log_counts = []
positive_benchmark_counts = []
negative_benchmark_counts = []
positive_log_sum = []
positive_benchmark_sum = []
positive_percent_sum = []
negative_log_sum = []
negative_benchmark_sum = []
negative_percent_sum = []
log_returns_sum = []  # This will store cumulative log returns for each simulation
percent_returns_sum = []
positive_log_returns_sum = []
negative_log_returns_sum = []
all_log_returns = []  # This will store daily log returns for each simulation
cumulative_means = []
cumulative_medians = []


# Simulation process
for _ in range(num_simulations):
    price = initial_price
    prices = [price]
    positive_log_days = 0
    negative_log_days = 0
    positive_log_change = 0
    negative_log_change = 0
    positive_log_returns = 0
    negative_log_returns = 0
    cumulative_log_returns = 0  # Initialize cumulative log return
    daily_log_returns = []  # Store daily log returns for this simulation
    cumulative_percentage_returns = 0
    daily_percentage_returns = []
    positive_percent_returns = 0
    negative_percent_returns = 0
    positive_benchmark_days = 0
    negative_benchmark_days = 0
    positive_benchmark_change = 0
    negative_benchmark_change = 0
    cumulative_benchmark_returns = 0

    for _ in range(num_days):
        if np.random.rand() <= probability_up:
            change_factor = 1 + volatility
        else:
            change_factor = 1 - volatility

        price *= change_factor
        prices.append(price)

        # Calculate daily log return
        log_returns = np.log(price / prices[-2])  # Using prices[-2] to access previous price
        cumulative_log_returns += log_returns  # Add to cumulative log return
        daily_log_returns.append(cumulative_log_returns)  # Append cumulative log return
        percent_returns = np.exp(log_returns) - 1
        cumulative_percentage_returns += percent_returns
        daily_percentage_returns.append(cumulative_percentage_returns)

        if log_returns >= 0:
            positive_log_days += 1
            positive_log_returns += log_returns
            positive_percent_returns += percent_returns
        elif log_returns < 0:
            negative_log_days += 1
            negative_log_returns += log_returns
            negative_percent_returns += percent_returns

        if price >= initial_price:
            positive_benchmark_days += 1
            positive_benchmark_change += (price - initial_price)
        else:
            negative_benchmark_days += 1
            negative_benchmark_change += (initial_price - price)

    all_prices.append(prices)
    positive_log_counts.append(positive_log_days)
    negative_log_counts.append(negative_log_days)
    positive_log_sum.append(positive_log_change)
    negative_log_sum.append(negative_log_change)
    log_returns_sum.append(cumulative_log_returns)  # Store cumulative log return
    percent_returns_sum.append(cumulative_percentage_returns)
    positive_log_returns_sum.append(positive_log_returns)
    negative_log_returns_sum.append(negative_log_returns)
    all_log_returns.append(daily_log_returns)  # Store daily cumulative log returns for this simulation
    positive_percent_sum.append(positive_percent_returns)
    negative_percent_sum.append(negative_percent_returns)
    positive_benchmark_counts.append(positive_benchmark_days)
    negative_benchmark_counts.append(negative_benchmark_days)
    positive_benchmark_sum.append(positive_benchmark_change)
    negative_benchmark_sum.append(negative_benchmark_change)

# Calculate the final prices for each simulation
final_prices = [prices[-1] for prices in all_prices]


# Plot Sample Means and Medians

# Loop through the final prices in steps of num_step
num_step = 100
for i in range(num_step, len(final_prices) + 1, num_step):
    current_mean = np.mean(final_prices[:i])
    current_median = np.median(final_prices[:i])
    cumulative_means.append(current_mean)
    cumulative_medians.append(current_median)


# Generate x-axis values (number of simulations used)
sample_size = np.arange(num_step, len(final_prices) + 1, num_step)

# Create a figure with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize as needed

# Plot cumulative means on the first subplot
axes[0].plot(sample_size, cumulative_means)
axes[0].set_xlabel(f'Sample Size (step ={num_step})')
axes[0].set_ylabel('Cumulative Average of Final Stock Price')
axes[0].set_title('Convergence of Average Final Price')
axes[0].grid(True)

# Plot cumulative medians on the second subplot
axes[1].plot(sample_size, cumulative_medians)
axes[1].set_xlabel(f'Sample Size (step ={num_step})')
axes[1].set_ylabel('Cumulative Median of Final Stock Price')
axes[1].set_title('Convergence of Median Final Price')
axes[1].grid(True)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# Calculate the statistics of the final prices
mean_final_prices = np.mean(final_prices)
median_final_prices = np.median(final_prices)
std_final_prices = np.std(final_prices)
max_final_prices = np.max(final_prices)
min_final_prices = np.min(final_prices)

# Calculate benchmark returns for each simulation
benchmark_returns = [(final_price / initial_price) - 1 for final_price in final_prices]


# Calculate the statistics of the log returns
mean_log_returns = np.mean(log_returns_sum)
median_log_returns = np.median(log_returns_sum)
std_log_returns = np.std(log_returns_sum)
total_positive_log_returns = np.sum(positive_log_returns_sum)
total_negative_log_returns = np.sum(negative_log_returns_sum)

max_log_returns = np.max(log_returns_sum)
min_log_returns = np.min(log_returns_sum)
mean_positive_log_returns = np.mean(positive_log_returns_sum)
mean_negative_log_returns = np.mean(negative_log_returns_sum)

mean_log_positive_count = np.mean(positive_log_counts)
mean_log_negative_count = np.mean(negative_log_counts)
total_log_positive_days = np.sum(positive_log_counts)
total_log_negative_days = np.sum(negative_log_counts)
positive_log_returns_per_day = mean_positive_log_returns / mean_log_positive_count
negative_log_returns_per_day = mean_negative_log_returns / mean_log_negative_count


# Calculate the statistics for percentage change
mean_percent_return = np.mean(percent_returns_sum)
median_percent_return = np.median(percent_returns_sum)
std_percent_return = np.std(percent_returns_sum)
mean_positive_percent_returns = np.mean(positive_percent_sum)
mean_negative_percent_returns = np.mean(negative_percent_sum)

# Calculate the statistics for benchmark returns
mean_benchmark_returns = np.mean(benchmark_returns)
median_benchmark_returns = np.median(benchmark_returns)
std_benchmark_returns = np.std(benchmark_returns)

# Calculate positive and negative benchmark returns
mean_positive_benchmark_days = np.mean(positive_benchmark_counts)
mean_negative_benchmark_days = np.mean(negative_benchmark_counts)
mean_positive_benchmark_sum = np.mean(positive_benchmark_sum)
mean_negative_benchmark_sum = np.mean(negative_benchmark_sum)
total_positive_benchmark_sum = np.sum(positive_benchmark_sum)
total_negative_benchmark_sum = np.sum(negative_benchmark_sum)
total_positive_benchmark_days = np.sum(positive_benchmark_counts)
total_negative_benchmark_days = np.sum(negative_benchmark_counts)
benchmark_positive_return_per_day = total_positive_benchmark_sum / total_positive_benchmark_days
benchmark_negative_return_per_day = total_negative_benchmark_sum / total_negative_benchmark_days

# Calculate skewness and kurtosis
log_returns_sum_array = np.array(log_returns_sum)
log_returns_kurtosis = kurtosis(log_returns_sum_array)
log_returns_skewness = skew(log_returns_sum_array)


# Print the results
print(f"Seed value: {seed_value}")
print(f"Initial price: {initial_price}")
print(f"Volatility: {volatility}")
print(f"Probability up: {probability_up}")
print(f"Number of days: {num_days}")
print(f"Number of simulations: {num_simulations}")
print(f"Mean final prices: {mean_final_prices}")
print(f"Median final prices: {median_final_prices}")
print(f"Std final prices/mean: {std_final_prices / mean_final_prices}")
print(f"Mean Log Returns: {mean_log_returns}")
print(f"Median Log Returns: {median_log_returns}")
print(f"Mean percentage return: {mean_percent_return * 100: .2f}%")
print(f"Median percentage return: {median_percent_return * 100: .2f}%")
print(f"Mean Benchmark Returns:{mean_benchmark_returns * 100: .2f}%")
print(f"Median Benchmark Returns:{median_benchmark_returns * 100:.2f}%")
print(f"Number of Positive Log Returns Days: {mean_log_positive_count}")
print(f"Number of Negative Log Returns Days: {mean_log_negative_count}")
print(f"Positive Log Returns per day:{mean_positive_log_returns / mean_log_positive_count}")
print(f"Negative Log Returns per day:{mean_negative_log_returns / mean_log_negative_count}")
print(f"Positive Percent Returns per day: {mean_positive_percent_returns / mean_log_positive_count}")
print(f"Negative Percent Returns per day: {mean_negative_percent_returns / mean_log_negative_count}")
print(f"Length of Percentage Returns: {len(percent_returns_sum)}")
print(f"Length of Log Returns: {len(log_returns_sum)}")
print(f"Number of Positive Benchmark days: {mean_positive_benchmark_days}")
print(f"Number of Negative Benchmark days: {mean_negative_benchmark_days}")
print(f"Positive Benchmark Return per Day: {benchmark_positive_return_per_day}")
print(f"Negative Benchmark Return per Day: {benchmark_negative_return_per_day}")


# Create a QQ figure with two subplots  for percent returns and benchmark returns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # (1 row, 2 columns). Adjust figsize as needed

# QQ-plot for percent returns on the first subplot
stats.probplot(percent_returns_sum, dist="norm", plot=axes[0])
axes[0].set_title("QQ-Plot of Daily Returns")

# QQ-plot for benchmark returns on the second subplot
stats.probplot(benchmark_returns, dist="norm", plot=axes[1])
axes[1].set_title("QQ-Plot Benchmark Returns")

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


#  Define sample_data
log_returns_sum_array = np.array(log_returns_sum)
percent_returns_sum_array = np.array(percent_returns_sum)
benchmark_returns_array = np.array(benchmark_returns)

#  Define positive sample_data
positive_log_returns_sum_array = log_returns_sum_array + abs(np.min(log_returns_sum_array)) + 1
positive_benchmark_returns_array = benchmark_returns_array + abs(np.min(benchmark_returns_array)) + 1
positive_percent_returns_sum_array = percent_returns_sum_array + abs(np.min(percent_returns_sum_array)) + 1

min_positive_benchmark = np.min(positive_benchmark_returns_array)
min_positive_percent = np.min(positive_percent_returns_sum_array)
print(f"min_positive_benchmark: {min_positive_benchmark}")
print(f"min_positive_percent: {min_positive_percent}")


def gini_coefficient(x: np.ndarray) -> float:
    """Compute the Gini coefficient of a numpy array."""
    # Check if the input is a numpy array
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    # Check if the input contains only non-negative values
    if np.any(x < 0):
        raise ValueError("Gini coefficient is only defined for non-negative values.")

    # Check if the input is not all zeros
    if np.all(x == 0):
        return 0.0

    # Sort the array in non-decreasing order
    x = np.sort(x)

    # Calculate the cumulative sum of the sorted array
    cumx = np.cumsum(x)

    # Calculate the Gini coefficient
    n = len(x)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    return gini


# Calculate the Gini coefficient for percent_returns_sum

gini_benchmark_returns = gini_coefficient(positive_benchmark_returns_array)
gini_percent_returns = gini_coefficient(positive_percent_returns_sum_array)
print(f"Gini coefficient for positive_benchmark_returns: {gini_benchmark_returns:.3f}")
print(f"Gini coefficient for positive_percent_returns: {gini_percent_returns:.3f}")


# Create Lorenz curves with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Lorenz curve for daily returns sum on the second subplot
axes[0].plot(np.linspace(0, 1, len(positive_percent_returns_sum_array)),
             np.cumsum(np.sort(positive_percent_returns_sum_array)) / np.sum(positive_percent_returns_sum_array),
             label='Lorenz curve')
axes[0].plot([0, 1], [0, 1], '--', label='45-degree line')
axes[0].fill_between(np.linspace(0, 1, len(positive_percent_returns_sum_array)),
                     np.cumsum(np.sort(positive_percent_returns_sum_array)) / np.sum(positive_percent_returns_sum_array),
                     color='blue', alpha=0.1)
axes[0].set_xlabel('Cumulative Share of Population')
axes[0].set_ylabel('Cumulative Share of Daily Returns')
axes[0].set_title('Daily Returns Lorenz Curve')
axes[0].text(0.6, 0.4, f'Gini = {gini_percent_returns:.3f}', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
axes[0].legend()

# Lorenz curve for benchmark returns on the first subplot
axes[1].plot(np.linspace(0, 1, len(positive_benchmark_returns_array)),
             np.cumsum(np.sort(positive_benchmark_returns_array)) / np.sum(positive_benchmark_returns_array),
             label='Lorenz curve')
axes[1].plot([0, 1], [0, 1], '--', label='45-degree line')
axes[1].fill_between(np.linspace(0, 1, len(positive_benchmark_returns_array)),
                     np.cumsum(np.sort(positive_benchmark_returns_array)) / np.sum(positive_benchmark_returns_array),
                     color='blue', alpha=0.1)
axes[1].set_xlabel('Cumulative Share of Population')
axes[1].set_ylabel('Cumulative Share of Benchmark Returns')
axes[1].set_title('Benchmark Returns Lorenz Curve')
axes[1].text(0.6, 0.4, f'Gini = {gini_benchmark_returns:.3f}', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
axes[1].legend()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


def hill_estimator(data):
    """
    - Calculates the Hill estimator for the tail index of a distribution.
    - Uses k = 0.1 * len(data) largest order statistics for estimation.
    - Args:
        data: A 1-dimensional NumPy array of positive data points.
    - Returns:
        The Hill estimator for the tail index.
    - Raises:
        ValueError: If the input data is not strictly positive.
    """

    # Input validation: Check for non-positive values
    if not np.all(data > 0):
        raise ValueError("Input data must contain only positive values for Hill estimator.")

    # Calculate k as % of the data length
    n = len(data)
    k = int(0.1 * n)

    # Extract the k largest order statistics
    sorted_data = np.sort(data)[::-1]  # Sort in descending order
    x_k = sorted_data[:k]

    # Calculate the Hill estimator
    return 1 / (np.mean(np.log(x_k / x_k[-1])))


tail_index_benchmark = hill_estimator(positive_benchmark_returns_array)
tail_index_percent_returns = hill_estimator(positive_percent_returns_sum_array)
print("Hill estimator benchmark returns:", tail_index_benchmark)
print("Hill estimator percent returns:", tail_index_percent_returns)


# Histograms

# Create Histograms with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize as needed

# Histogram for percent returns on the first subplot
counts1, bins1, _ = axes[0].hist(percent_returns_sum, bins=39, edgecolor='black', alpha=0.7, rwidth=.99)
frequencies1 = counts1 / num_simulations * 100  # Calculate frequencies as percentages
axes[0].clear()  # Clear the current plot
axes[0].bar(bins1[:-1], frequencies1, width=np.diff(bins1), edgecolor='black', alpha=0.7, align='edge')  # Plot with percentages
axes[0].set_title('Histogram of Daily Returns')
axes[0].set_xlabel('Daily Returns')
axes[0].set_ylabel('Frequency (%)')

# Histogram for benchmark returns on the second subplot
counts2, bins2, _ = axes[1].hist(benchmark_returns, bins=39, edgecolor='black', alpha=0.7, rwidth=.99)
frequencies2 = counts2 / num_simulations * 100  # Calculate frequencies as percentages
axes[1].clear()  # Clear the current plot
axes[1].bar(bins2[:-1], frequencies2, width=np.diff(bins2), edgecolor='black', alpha=0.7, align='edge')  # Plot with percentages
axes[1].set_title('Histogram of Benchmark Returns')
axes[1].set_xlabel('Benchmark Returns')
axes[1].set_ylabel('Frequency (%)')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Log-log plots for Daily and Benchmark returns

# Sort the data for Daily returns
sorted_daily_data = np.sort(positive_percent_returns_sum_array)
ecdf_daily = np.arange(1, len(sorted_daily_data) + 1) / len(sorted_daily_data)

# Sort the data for Benchmark returns
sorted_benchmark_data = np.sort(positive_benchmark_returns_array)
ecdf_benchmark = np.arange(1, len(sorted_benchmark_data) + 1) / len(sorted_benchmark_data)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for Daily returns
axes[0].loglog(sorted_daily_data, 1 - ecdf_daily, marker='.', linestyle='none', markersize=5)
axes[0].set_xlabel('Log(Data)')
axes[0].set_ylabel('Log(1 - ECDF)')
axes[0].set_title('Log-Log Plot of Daily Returns')
axes[0].grid(True, which="both", ls="--")

# Plot for Benchmark returns
axes[1].loglog(sorted_benchmark_data, 1 - ecdf_benchmark, marker='.', linestyle='none', markersize=5)
axes[1].set_xlabel('Log(Data)')
axes[1].set_ylabel('Log(1 - ECDF)')
axes[1].set_title('Log-Log Plot of Benchmark Returns')
axes[1].grid(True, which="both", ls="--")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()



# Calculate the time of computation
end_time = time.time()
print(f"Elapsed time (seconds): {end_time - start_time}")
