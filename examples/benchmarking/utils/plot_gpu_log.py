import polars as pl
import matplotlib.pyplot as plt
import sys

def plot_gpu_metrics(csv_file):
    df = pl.read_csv(csv_file)

    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )

    gpu_indices = df.select("gpu_index").unique().to_series().to_list()

    metrics = ["gpu_util", "mem_used", "power", "temp"]
    metric_labels = {
        "gpu_util": "GPU Utilization (%)",
        "mem_used": "Memory Used (MB)",
        "power": "Power Usage (W)",
        "temp": "Temperature (°C)"
    }

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for gpu in gpu_indices:
            gpu_df = df.filter(pl.col("gpu_index") == gpu)
            plt.plot(gpu_df["timestamp"], gpu_df[metric], label=f"GPU {gpu}")

        plt.xlabel("Time")
        plt.ylabel(metric_labels[metric])
        plt.title(f"{metric_labels[metric]} Over Time")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.savefig(f"{metric}{csv_file}_plot.png")
        print(f"[+] Saved: {metric}_plot.png")
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot_gpu_log.py gpu_log_YYYYMMDD_HHMMSS.csv")
        sys.exit(1)

    plot_gpu_metrics(sys.argv[1])
