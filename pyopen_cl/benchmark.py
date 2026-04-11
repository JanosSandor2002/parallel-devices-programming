import time
import json
import numpy as np
import pyopencl as cl

from config import N, KERNEL_FILE, RESULT_JSON
from data import generate_data
from backend import get_device, build_context, load_kernel, run_kernel


def banner(text: str) -> None:
    line = "─" * 50
    print(f"\n{line}\n  {text}\n{line}")


def check_result(cpu_val: float, gpu_val: float, tol: float = 1e-2) -> bool:
    rel_err = abs(cpu_val - gpu_val) / abs(cpu_val)
    ok = rel_err < tol
    status = "✓ OK" if ok else "✗ ELTÉRÉS"
    print(f"  Relatív eltérés: {rel_err:.2e}  →  {status}")
    return ok


def main():
    # Adatok

    banner("Adatok előkészítése")

    x = generate_data()

    print(f"  N          = {N:,}")
    print(f"  dtype      = {x.dtype}")
    print(f"  min / max  = {x.min():.4f} / {x.max():.4f}")
    print(f"  memória    ≈ {x.nbytes / 1024**2:.1f} MB")


    # CPU

    banner("CPU számítás (NumPy)")

    _ = np.sum(x ** 2)  # warmup

    t0 = time.perf_counter()
    cpu_result = np.sum(x ** 2)
    cpu_time = time.perf_counter() - t0

    print(f"  Eredmény   = {cpu_result:.6f}")
    print(f"  Idő        = {cpu_time * 1000:.2f} ms")


    # GPU init

    banner("OpenCL (GPU) inicializálás")

    device = get_device()
    ctx, queue = build_context(device)
    program = load_kernel(ctx, KERNEL_FILE)

    print(f"  Platform   : {device.platform.name}")
    print(f"  Eszköz     : {device.name}")
    print(f"  Max CU     : {device.max_compute_units}")
    print(f"  Globális mem: {device.global_mem_size / 1024**3:.1f} GB")


    mf = cl.mem_flags
    input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, x.nbytes)


    # GPU

    banner("GPU számítás (OpenCL kernel)")

    program.sum_of_squares(queue, (N,), None, input_buf, output_buf)
    queue.finish()

    t0 = time.perf_counter()

    event = run_kernel(queue, program, N, input_buf, output_buf)

    squared_gpu = np.empty_like(x)
    cl.enqueue_copy(queue, squared_gpu, output_buf)
    queue.finish()

    gpu_time_full = time.perf_counter() - t0

    kernel_ns = event.profile.end - event.profile.start
    kernel_time = kernel_ns / 1e6

    gpu_result = float(np.sum(squared_gpu))

    print(f"  Eredmény             = {gpu_result:.6f}")
    print(f"  Kernel idő           = {kernel_time:.2f} ms")
    print(f"  Teljes pipeline idő  = {gpu_time_full * 1000:.2f} ms")


    # Ellenőrzés

    banner("Helyesség-ellenőrzés")

    cpu_squared = x ** 2
    max_diff = float(np.max(np.abs(cpu_squared - squared_gpu)))

    print(f"  Max elem-szintű eltérés: {max_diff:.2e}")
    check_result(cpu_result, gpu_result)

    # Speedup

    banner("Teljesítmény összefoglalás")

    speedup_full = cpu_time / gpu_time_full
    speedup_kernel = (cpu_time * 1000) / kernel_time

    print(f"  CPU idő              : {cpu_time * 1000:.2f} ms")
    print(f"  GPU kernel idő       : {kernel_time:.2f} ms")
    print(f"  GPU teljes pipeline  : {gpu_time_full * 1000:.2f} ms")
    print(f"  Gyorsítás (kernel)   : {speedup_kernel:.1f}×")
    print(f"  Gyorsítás (pipeline) : {speedup_full:.1f}×")


    # Mentés

    results = {
        "N": N,
        "cpu_time_ms": cpu_time * 1000,
        "gpu_kernel_ms": kernel_time,
        "gpu_pipeline_ms": gpu_time_full * 1000,
        "cpu_result": float(cpu_result),
        "gpu_result": float(gpu_result),
        "speedup_kernel": speedup_kernel,
        "speedup_pipeline": speedup_full,
    }

    with open(RESULT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Eredmények mentve → {RESULT_JSON}")
    print("\n  Futtasd a grafikonhoz: python plot_results.py\n")


if __name__ == "__main__":
    main()