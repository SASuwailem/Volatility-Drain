import numpy as np
import scipy.stats as stats
import random
import matplotlib.pyplot as plt


# Setup parameters
seed_value = 489266  # random.randint(0, 1000000)
random.seed(seed_value)
np.random.seed(seed_value)
sample_size = 10000
Hill_k = 0.1
print(f"Seed number:{seed_value}")
print(f"Sample size: {sample_size}")
print(f"Hill estimator k = {Hill_k*100}%")

# Define the Hill estimator function


def hill_estimator(data):
    """
    Calculates the Hill estimator for the tail index of a distribution.
    Uses k = 0.1 * len(data) the largest order statistics for estimation.

    Args:
        data: A 1-dimensional NumPy array of positive data points.

    Returns:
        The Hill estimator for the tail index.

    Raises:
        ValueError: If the input data is not strictly positive.
    """

    # Input validation: Check for non-positive values
    if not np.all(data > 0):
        raise ValueError("Input data must contain only positive values for Hill estimator.")

    # Calculate k as % of the data length
    n = len(data)
    k = int(Hill_k * n)

    # Extract the k largest order statistics
    sorted_data = np.sort(data)[::-1]  # Sort in descending order
    x_k = sorted_data[:k]

    # Calculate the Hill estimator
    return 1 / (np.mean(np.log(x_k / x_k[-1])))


# Calculate the Hill estimator for different distributions

# (i) Normal distribution (light-tailed)
normal_mean = 0
normal_std = 1
normal_data = np.random.normal(loc=normal_mean, scale=normal_std, size=sample_size)

# Ensure all sample values are positive
normal_data_positive = normal_data + abs(np.min(normal_data)) + 1
print(f"Minimum normal positive sample: {np.min(normal_data_positive)}")

# Hill estimator for normal distribution
hill_normal = hill_estimator(normal_data_positive)
print(f"Hill estimator for Normal distribution ({normal_mean}, {normal_std}): {hill_normal:.4f}")

# Sort the data for the Normal distribution
sorted_normal_data = np.sort(normal_data_positive)
ecdf_normal = np.arange(1, sample_size + 1) / sample_size

# (ii) Pareto distribution (heavy-tailed)
alpha = 2  # Shape parameter
pareto_data = stats.pareto.rvs(b=alpha, size=sample_size)
hill_pareto = hill_estimator(pareto_data)
print(f"Hill estimator for Pareto distribution (alpha = {alpha}): {hill_pareto:.4f}")

# Sort the data for the Pareto distribution
sorted_pareto_data = np.sort(pareto_data)
ecdf_pareto = np.arange(1, sample_size + 1) / sample_size

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot for the Normal distribution
axes[0].loglog(sorted_normal_data, 1 - ecdf_normal, marker='.', linestyle='none', markersize=5)
axes[0].set_xlabel('Log(Data)')
axes[0].set_ylabel('Log(1 - ECDF)')
axes[0].set_title('Log-Log Plot of Normal-Distributed Data')
axes[0].grid(True, which="both", ls="--")

# Plot for the Pareto distribution
axes[1].loglog(sorted_pareto_data, 1 - ecdf_pareto, marker='.', linestyle='none', markersize=5)
axes[1].set_xlabel('Log(Data)')
axes[1].set_ylabel('Log(1 - ECDF)')
axes[1].set_title('Log-Log Plot of Pareto-Distributed Data')
axes[1].grid(True, which="both", ls="--")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


# (iii) Student's t-distribution (heavy-tailed)
df = 1.5  # Degrees of freedom
t_data = stats.t.rvs(df=df, size=sample_size)
# Ensure all values are positive
t_data = np.abs(t_data)
hill_t = hill_estimator(t_data)
print(f"Hill estimator for Student's t-distribution (df = {df}): {hill_t:.4f}")

# Plot log-log graph for the Student distribution
sorted_t_data = np.sort(t_data)
ecdf_t = np.arange(1, sample_size + 1) / sample_size

plt.figure(figsize=(8, 6))
plt.loglog(sorted_t_data, 1 - ecdf_t, marker='.', linestyle='none', markersize=5)
plt.xlabel('Log(Data)')
plt.ylabel('Log(1 - ECDF)')
plt.title('Log-Log Plot of Student-Distributed Data')
plt.grid(True, which="both", ls="--")
plt.show()


# (iv) Weibull distribution (heavy-tailed, shape < 1)
shape = .5  # Shape parameter (k < 1 for heavy-tailed)
weibull_data = stats.weibull_min.rvs(c=shape, size=sample_size)
hill_weibull = hill_estimator(weibull_data)
print(f"Hill estimator for Weibull distribution (shape = {shape}): {hill_weibull:.4f}")

# Plot log-log graph for the Weibull distribution
sorted_weibull_data = np.sort(weibull_data)
ecdf_weibull = np.arange(1, sample_size + 1) / sample_size

plt.figure(figsize=(8, 6))
plt.loglog(sorted_weibull_data, 1 - ecdf_weibull, marker='.', linestyle='none', markersize=5)
plt.xlabel('Log(Data)')
plt.ylabel('Log(1 - ECDF)')
plt.title('Log-Log Plot of Weibull-Distributed Data')
plt.grid(True, which="both", ls="--")
plt.show()


# (v) Cauchy distribution (heavy-tailed)
cauchy_data = stats.cauchy.rvs(size=sample_size)
# Ensure all values are positive (Cauchy is symmetric, so we can take absolute values)
cauchy_data = np.abs(cauchy_data)
hill_cauchy = hill_estimator(cauchy_data)
print(f"Hill estimator for Cauchy distribution: {hill_cauchy:.4f}")

# Plot log-log graph for the Cauchy distribution
sorted_cauchy_data = np.sort(cauchy_data)
ecdf_cauchy = np.arange(1, sample_size + 1) / sample_size

plt.figure(figsize=(8, 6))
plt.loglog(sorted_cauchy_data, 1 - ecdf_cauchy, marker='.', linestyle='none', markersize=5)
plt.xlabel('Log(Data)')
plt.ylabel('Log(1 - ECDF)')
plt.title('Log-Log Plot of Cauchy-Distributed Data')
plt.grid(True, which="both", ls="--")
plt.show()