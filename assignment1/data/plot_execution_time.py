import json
import numpy as np
import matplotlib.pyplot as plt

FILE = "times_thread.json"
FILE = "times_openmp.json"
TITLE = "Speedup - Thread"
TITLE = "Speedup - OpenMP"
COLORS = ['#CB4335', '#2471A3', '#1E8449', '#CA6F1E', '#1ABC9C', '#884EA0', '#2E4053']
TIME = 'time'
OUT = 'out_thread.svg'
OUT = 'out_openmp.svg'
DATA_NAMES = ['brightkite', 'facebook', 'full_graph', 'gemsec-facebook', 'github', 'gowalla']
LIMIT = 10


def main():
    # Open the JSON file
    with open(FILE, 'r') as f:
        # Load the JSON data from the file
        data = json.load(f)

    for data_name, c in zip(DATA_NAMES, COLORS):
        times = data[data_name][TIME]
        sequential_time = times["1"]

        cores = list(times.keys())[:LIMIT]
        cores = [int(c) for c in list(times.keys())[:LIMIT]]
        speedup = [sequential_time / t for t in list(times.values())][:LIMIT]

        plt.plot(cores, speedup, linestyle='solid', label=data_name, color=c)

    plt.plot(cores, cores, linestyle='dashed', label="Linear speedup", color=COLORS[-1])
    plt.fill_between(cores, cores, color='red', alpha=0.2, label="Sublinear")
    plt.fill_between([c for c in cores], [10 for c in cores], color='green', alpha=0.1, label="Superlinear")

    # Add a legend
    plt.legend()

    # Set the x and y axis labels
    plt.title(TITLE)
    plt.xlabel('N Threads')
    plt.ylabel('Speedup')

    plt.savefig(OUT)


if __name__ == "__main__":
    main()
