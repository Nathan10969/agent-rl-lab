"""Helpers for measuring the GPU selected by the actual CUDA process."""
import subprocess
import sys
import threading
import time


def find_gpu_uuid(rows, pid):
    for row in rows.splitlines():
        fields = [value.strip() for value in row.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            return fields[1]
    return None


def parse_gpu_memory(row):
    fields = [value.strip() for value in row.strip().split(",")]
    if len(fields) != 2:
        raise ValueError("expected used and total GPU memory in MiB")
    return int(fields[0]), int(fields[1])


def summarize_memory(samples):
    if not samples:
        return None
    return {"initial_used_mib": samples[0][0], "sampled_peak_used_mib": max(used for used, _ in samples),
            "total_mib": samples[0][1], "samples": len(samples)}


def identify_allocated_gpu():
    """Use a short-lived CUDA process to map Slurm's device to an NVML UUID."""
    code = 'import time, torch; x = torch.ones(1, device="cuda"); torch.cuda.synchronize(); time.sleep(30)'
    helper = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline and helper.poll() is None:
            rows = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                text=True, timeout=5)
            uuid = find_gpu_uuid(rows, helper.pid)
            if uuid:
                return uuid
            time.sleep(0.5)
        return None
    finally:
        if helper.poll() is None:
            helper.terminate()
            try:
                helper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                helper.kill()
                helper.wait()


class GPUMonitor:
    """Sample the identified GPU every second; sampled maxima are not instantaneous peaks."""
    def __init__(self, uuid):
        self.uuid = uuid
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                row = subprocess.check_output(
                    ["nvidia-smi", "--id=" + self.uuid, "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"], text=True, timeout=5)
                self.samples.append(parse_gpu_memory(row))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.errors.append(str(error))
            self.stop_event.wait(1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
