import os
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
    # ── Adatok ───────────────────────────────────────────────
    banner("Adatok előkészítése")
    x = generate_data()
    print(f"  N          = {N:,}")
    print(f"  dtype      = {x.dtype}")
    print(f"  min / max  = {x.min():.4f} / {x.max():.4f}")
    print(f"  memória    ≈ {x.nbytes / 1024**2:.1f} MB")

    # ── CPU ──────────────────────────────────────────────────
    banner("CPU számítás (NumPy)")
    _ = np.sum(x ** 2)   # warmup
    t0 = time.perf_counter()
    cpu_result = float(np.sum(x ** 2))
    cpu_time = time.perf_counter() - t0
    print(f"  Eredmény   = {cpu_result:.6f}")
    print(f"  Idő        = {cpu_time * 1000:.2f} ms")

    # ── GPU init ─────────────────────────────────────────────
    banner("OpenCL (GPU) inicializálás")
    device = get_device()
    ctx, queue = build_context(device)
    kernels = load_kernel(ctx, KERNEL_FILE)   # <-- tuple (knl_sum, knl_finish)
    print(f"  Platform   : {device.platform.name}")
    print(f"  Eszköz     : {device.name}")
    print(f"  Max CU     : {device.max_compute_units}")
    print(f"  Globális mem: {device.global_mem_size / 1024**3:.1f} GB")

    mf = cl.mem_flags
    input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

    # ── GPU TEST MODE (debug reduction trace) ─────────────────────────
    banner("GPU TESZT (stride / partial reduction trace)")

    debug_result_buf, debug_events = run_kernel(
        queue,
        kernels,
        N,
        input_buf,
        debug=True   # <- ezt kell hozzáadni a run_kernelhez
    )

    debug_arr = np.empty(1, dtype=np.float32)
    cl.enqueue_copy(queue, debug_arr, debug_result_buf)
    queue.finish()

    print(f"\n  DEBUG FINAL RESULT = {debug_arr[0]}\n")

    # ── GPU warmup ───────────────────────────────────────────
    result_buf_w, _ = run_kernel(queue, kernels, N, input_buf)   # <-- kernels
    queue.finish()

    # ── GPU számítás ─────────────────────────────────────────
    banner("GPU számítás (OpenCL kernel – teljes reduction GPU-n)")

    t0 = time.perf_counter()
    result_buf, events = run_kernel(queue, kernels, N, input_buf)   # <-- kernels

    # Egyetlen float visszaolvasása a GPU-ról
    gpu_result_arr = np.empty(1, dtype=np.float32)
    cl.enqueue_copy(queue, gpu_result_arr, result_buf)
    queue.finish()

    gpu_time_full = time.perf_counter() - t0
    gpu_result = float(gpu_result_arr[0])

    # Kernel idők (profiling)
    kernel_ns = sum(ev.profile.end - ev.profile.start for ev in events)
    kernel_time = kernel_ns / 1e6

    print(f"  Eredmény             = {gpu_result:.6f}")
    print(f"  Kernel idő (2 pass)  = {kernel_time:.2f} ms")
    print(f"  Teljes pipeline idő  = {gpu_time_full * 1000:.2f} ms")

    # ── Ellenőrzés ───────────────────────────────────────────
    banner("Helyesség-ellenőrzés")
    check_result(cpu_result, gpu_result)

    # ── Speedup ──────────────────────────────────────────────
    banner("Teljesítmény összefoglalás")
    speedup_full   = cpu_time / gpu_time_full
    speedup_kernel = (cpu_time * 1000) / kernel_time

    print(f"  CPU idő              : {cpu_time * 1000:.2f} ms")
    print(f"  GPU kernel idő       : {kernel_time:.2f} ms")
    print(f"  GPU teljes pipeline  : {gpu_time_full * 1000:.2f} ms")
    print(f"  Gyorsítás (kernel)   : {speedup_kernel:.1f}×")
    print(f"  Gyorsítás (pipeline) : {speedup_full:.1f}×")

    # ── Mentés ───────────────────────────────────────────────
    results = {
        "N": N,
        "cpu_time_ms":      cpu_time * 1000,
        "gpu_kernel_ms":    kernel_time,
        "gpu_pipeline_ms":  gpu_time_full * 1000,
        "cpu_result":       cpu_result,
        "gpu_result":       gpu_result,
        "speedup_kernel":   speedup_kernel,
        "speedup_pipeline": speedup_full,
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Eredmények mentve → {RESULT_JSON}")
    print("\n  Futtasd a grafikonhoz: python results.py\n")


if __name__ == "__main__":
    main()