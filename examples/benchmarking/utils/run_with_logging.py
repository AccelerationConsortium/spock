import subprocess
import time
import multiprocessing
import polars as pl
from gpu_profiling import collect_gpu_data

def run_benchmark():
    print("[+] Running benchmark script...")
    subprocess.run(["python3", "benchmark_llama3.3.py"])
    print("[+] Benchmark completed.")

def main():
    manager = multiprocessing.Manager()
    shared_list = manager.list()

    logger_proc = multiprocessing.Process(
        target=collect_gpu_data,
        args=(shared_list,),
        kwargs={"interval": 1}
    )
    logger_proc.start()
    print(f"[+] GPU logger started (PID {logger_proc.pid})")

    run_benchmark()

    print("[+] Stopping GPU logger...")
    logger_proc.terminate()
    logger_proc.join()

    df = pl.DataFrame(list(shared_list))  
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"gpu_log_{timestamp}.csv"
    df.write_csv(output_file)

    print(f"[+] GPU log written to: {output_file}")
if __name__ == "__main__":
    main()
