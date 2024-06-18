import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.stats import norm
from matplotlib.patches import Polygon

# Constants
ORDER = 6
SEED = 2212894100  # np.random.randint(0, 2**32)


def sierpinski(order, points):
    """Recursively generates the points for the Sierpinski triangle of a given order."""
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer")
    if len(points) != 3:
        raise ValueError("Points must be a list of three tuples")

    if order == 0:
        return [points]

    mid = lambda p1, p2: ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    p1, p2, p3 = points
    mid1 = mid(p1, p2)
    mid2 = mid(p2, p3)
    mid3 = mid(p3, p1)

    triangles = []
    triangles.extend(sierpinski(order - 1, [p1, mid1, mid3]))
    triangles.extend(sierpinski(order - 1, [mid1, p2, mid2]))
    triangles.extend(sierpinski(order - 1, [mid3, mid2, p3]))

    return triangles


def plot_sierpinski(triangles):
    """Plots the generated Sierpinski triangles."""
    fig, ax = plt.subplots()
    for triangle in triangles:
        polygon = Polygon(triangle, closed=True, edgecolor='b')
        ax.add_patch(polygon)
    ax.set_aspect('equal')
    plt.title(f'Sierpinski Triangle of Order {ORDER}')
    plt.show()


def count_triangles(order):
    """Counts the number of triangles of each size in a Sierpinski triangle of a given order."""
    counts = defaultdict(int)
    for i in range(order + 1):
        num_triangles = 3 ** i  # Number of triangles at each level
        size = 1 / (2 ** i)  # Size of triangles at each level
        counts[size] = num_triangles
    return counts


def plot_histograms(sizes_data, titles, colors, bins, min_size, max_size):
    """Plots count histograms for multiple datasets."""
    fig, axes = plt.subplots(1, len(sizes_data), figsize=(12, 6))
    for ax, sizes, title, color in zip(axes, sizes_data, titles, colors):
        ax.hist(sizes, bins=bins, color=color, alpha=0.7, edgecolor='black', density=False)
        ax.set_xlabel('Size of Triangle')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xlim(min_size, max_size)
    plt.tight_layout()
    plt.show()


def plot_frequency_histograms(sizes_data, titles, colors, bins, min_size, max_size):
    """Plots frequency histograms for multiple datasets."""
    fig, axes = plt.subplots(1, len(sizes_data), figsize=(12, 6))
    for ax, sizes, title, color in zip(axes, sizes_data, titles, colors):
        counts, bin_edges = np.histogram(sizes, bins=bins)
        total = np.sum(counts)
        frequencies = counts / total  # Calculate frequencies
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
        ax.bar(bin_centers, frequencies, width=(bin_edges[1] - bin_edges[0]), color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Size of Triangle')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.set_xlim(min_size, max_size)
    plt.tight_layout()
    plt.show()


def generate_random_triangles_normal(num_triangles, mean, sigma, min_size, max_size):
    """Generates random triangles with sizes following a normal distribution."""
    normal_dist = norm(loc=mean, scale=sigma)
    clipped_sizes = []
    while len(clipped_sizes) < num_triangles:
        additional_sizes = normal_dist.rvs(num_triangles - len(clipped_sizes))
        clipped_sizes.extend(additional_sizes[(additional_sizes >= min_size) & (additional_sizes <= max_size)])
    return clipped_sizes


def main():
    # Set the seed for reproducibility
    np.random.seed(SEED)

    # Generate points for the initial large triangle
    initial_triangle = [(0, 0), (1, 0), (0.5, (3 ** 0.5) / 2)]

    # Generate the Sierpinski triangles
    triangles = sierpinski(ORDER, initial_triangle)
    # print(f"Generated {len(triangles)} triangles")

    # Plot the Sierpinski triangle
    plot_sierpinski(triangles)

    # Count triangles by size for Sierpinski
    sierpinski_counts = count_triangles(ORDER)

    # Print counts for debugging
    print("Sierpinski Counts by Size:")
    for size, count in sierpinski_counts.items():
        print(f"Size: {size}, Count: {count}")

    # Convert counts dictionary to a list of sizes for histogram plotting
    sizes_sierpinski = []
    total_count = 0
    for size, count in sierpinski_counts.items():
        sizes_sierpinski.extend([size] * count)
        total_count += count
        print(f"Adding {count} triangles of size {size}, Total Count: {total_count}")

    # Ensure the total number of triangles is as expected
    expected_total = (3 ** (ORDER + 1) - 1) // 2
    assert len(
        sizes_sierpinski) == expected_total, f"Expected {expected_total} triangles, but got {len(sizes_sierpinski)}"

    # Determine min and max sizes from the Sierpinski triangle
    min_size = min(sierpinski_counts.keys())
    max_size = max(sierpinski_counts.keys())

    # Set the mean to the midpoint and adjust sigma accordingly
    mean_normal = (min_size + max_size) / 2
    sigma_normal = (max_size - min_size) / 4  # Assuming 95% of values should lie within min and max

    # Generate random triangles and count by size using a normal distribution
    sizes_normal = generate_random_triangles_normal(len(sizes_sierpinski), mean_normal, sigma_normal, min_size,
                                                    max_size)

    # Define bins for histograms using linear space
    bins = np.linspace(min_size, max_size, 20)

    # Plot the histograms for comparison (count)
    plot_histograms([sizes_sierpinski, sizes_normal],
                    ['Sierpinski Triangle Histogram (Count)', 'Random Triangles Histogram (Count)'],
                    ['blue', 'purple'],
                    bins, min_size, max_size)

    # Plot the histograms for comparison (frequency)
    plot_frequency_histograms([sizes_sierpinski, sizes_normal],
                              ['Sierpinski Triangle Histogram (Frequency)', 'Random Triangles Histogram (Frequency)'],
                              ['blue', 'purple'],
                              bins, min_size, max_size)

    # Print the seed value after execution
    print(f"Random seed used: {SEED}")
    print(f"Order of Sierpinski Triangle: {ORDER}")
    print(f"Number of triangles: {expected_total}")
    print(f"Minimum triangle size: {min_size}")
    print(f"Maximum triangle size: {max_size}")


if __name__ == "__main__":
    main()
