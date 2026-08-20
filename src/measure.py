import time
from pathlib import Path
from typing import Any

def memorySize(fileName: str) -> float:
    mem_mo = Path(fileName).stat().st_size / 1e6
    return mem_mo

def latency(model: Any, image: Any) -> float:
    t0 = time.perf_counter()
    _ = model(image)
    latency_ms = (time.perf_counter() - t0) * 1000
    return latency_ms