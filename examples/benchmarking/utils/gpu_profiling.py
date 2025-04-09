import pynvml
import time
import polars as pl
from multiprocessing import Manager

def collect_gpu_data(shared_list, interval=1, pid=None):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            used_by_pid = None
            if pid is not None:
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in procs:
                        if p.pid == pid:
                            used_by_pid = round(p.usedGpuMemory / 1024**2, 2)
                except pynvml.NVMLError:
                    pass

            shared_list.append({
                "timestamp": timestamp,
                "gpu_index": i,
                "gpu_util": util.gpu,
                "mem_used": round(mem_info.used / 1024**2, 2),
                "power": power,
                "temp": temp,
                "pid": pid,
                "used_mem_by_pid": used_by_pid
            })

        time.sleep(interval)
