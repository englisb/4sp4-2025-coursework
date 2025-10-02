import json
import matplotlib.pyplot as plt
import os

def plot_google_benchmark_median(json_path):
    # TODO: open the json file and do necessary processing to plot what is needed.

    plt.figure(figsize=(10, 6))

    plt.ylabel("y-label")
    plt.xlabel("x-label")
    plt.title("Google Benchmark Median Times")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # plots folder does not exist, create it
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    plt.savefig("./plots/plot1.png")

# Example usage: the input should stay the same. The output plot should be saved in the plots folder.
plot_google_benchmark_median("./logs/lab02.json")