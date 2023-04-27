import os
import os.path as path
import subprocess
import re
import json

FILE_EXECUTION = ["main_openmp", "main_thread"]
INFO_FILE = 'info.json'
PATTERN = r"Correct - Execution time: (\d+\.\d+) sec"

THREAD_START = 1
THREAD_END = 30

DATA_DIR = "./../data"
OUT_NAMES = ["times_openmp.json", "times_thread.json"]
OUT_NAMES = ["densities.json", "densities.json"]

NODES = 'nodes'
EDGES = 'edges'
TRIANGLES = 'triangles'
PERFORMANCE = 'time'
DENSITY = 'density'


def launch(exec_file:str, data_name: str, n_threads: int) -> int:
    cmd_line = [f"./{exec_file}", data_name, str(n_threads)]
    out = subprocess.run(cmd_line, stdout=subprocess.PIPE).stdout.decode('utf-8')
    last_line = out.splitlines()[-2]
    match = re.search(PATTERN, last_line)
    if match:
        return float(match.group(1))  
    else:
        raise Exception(f"{data_name}: {last_line}")


def main():
    data_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"Processing datasets: {data_names}")
    for exe, out_name in zip(FILE_EXECUTION, OUT_NAMES):
        out_ = dict()
        print(f"Version: {exe}")
        for data_name in data_names:
            print(f" > Processing {data_name}")
            info_file = path.join(DATA_DIR, data_name, INFO_FILE)
            with open(info_file, 'r') as f:
                info = json.load(f)
                out_[data_name] = {
                    NODES : info[NODES],
                    EDGES : info[EDGES],
                    DENSITY: 2 * info[EDGES] / ( info[NODES] * (info[NODES] - 1) ),
                    PERFORMANCE : {
                        n_threads : 
                        launch(
                            exec_file=exe,
                            data_name=data_name,
                            n_threads=n_threads
                        )
                        for n_threads in range(THREAD_START, THREAD_END+1)
                    }
                }
        out_file = path.join(DATA_DIR, out_name)    
        with open(out_file, "w") as f:
            json.dump(out_, f)

if __name__ == "__main__":
    main()