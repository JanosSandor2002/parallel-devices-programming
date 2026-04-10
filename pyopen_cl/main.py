import os
import sys
import time
import numpy as np
import pyopencl as cl

# Konfiguráció
N          = 10_000_000   # Tömb mérete
SEED       = 42
KERNEL_FILE = os.path.join(os.path.dirname(__file__), "sum_of_squares.cl")

# Segédfüggvények
def _banner(text: str) -> None:
    line = "─" * 50
    print(f"\n{line}\n  {text}\n{line}")


def _check_result(cpu_val: float, gpu_val: float, tol: float = 1e-2) -> bool:
    """Relatív eltérés ellenőrzése float32 pontossággal."""
    rel_err = abs(cpu_val - gpu_val) / abs(cpu_val)
    ok = rel_err < tol
    status = "✓ OK" if ok else "✗ ELTÉRÉS"
    print(f"  Relatív eltérés: {rel_err:.2e}  →  {status}")
    return ok


# Adatok előkészítése
_banner("Adatok előkészítése")
rng = np.random.default_rng(SEED)
x = rng.random(N).astype(np.float32)

print(f"  N          = {N:,}")
print(f"  dtype      = {x.dtype}")
print(f"  min / max  = {x.min():.4f} / {x.max():.4f}")
print(f"  memória    ≈ {x.nbytes / 1024**2:.1f} MB")


# CPU – NumPy
_banner("CPU számítás (NumPy)")

# Melegítés (első hívás cache-elheti az allokációt)
_ = np.sum(x ** 2)

t0 = time.perf_counter()
cpu_result = np.sum(x ** 2)
cpu_time   = time.perf_counter() - t0

print(f"  Eredmény   = {cpu_result:.6f}")
print(f"  Idő        = {cpu_time * 1000:.2f} ms")


# OpenCL – GPU
_banner("OpenCL (GPU) inicializálás")

# Platform és eszköz kiválasztása
platforms = cl.get_platforms()
if not platforms:
    print("  [HIBA] Nem található OpenCL platform.", file=sys.stderr)
    sys.exit(1)

# GPU preferált, fallback CPU
gpu_device = None
for plat in platforms:
    gpus = plat.get_devices(device_type=cl.device_type.GPU)
    if gpus:
        gpu_device = gpus[0]
        break

if gpu_device is None:
    print("  [FIGYELEM] GPU nem elérhető, CPU OpenCL eszközt használok.")
    gpu_device = platforms[0].get_devices()[0]

print(f"  Platform   : {gpu_device.platform.name}")
print(f"  Eszköz     : {gpu_device.name}")
print(f"  Max CU     : {gpu_device.max_compute_units}")
print(f"  Globális mem: {gpu_device.global_mem_size / 1024**3:.1f} GB")

ctx   = cl.Context(devices=[gpu_device])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Kernel betöltés,fordítása
with open(KERNEL_FILE, "r") as fh:
    source = fh.read()

program = cl.Program(ctx, source).build()

# Buffer-ek foglalása
mf         = cl.mem_flags
input_buf  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=x)
output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, x.nbytes)


# GPU futtatás (idő mérés: teljes pipeline)
_banner("GPU számítás (OpenCL kernel)")

# Melegítés
program.sum_of_squares(queue, (N,), None, input_buf, output_buf)
queue.finish()

# Mérés
t0 = time.perf_counter()

event = program.sum_of_squares(queue, (N,), None, input_buf, output_buf)

squared_gpu = np.empty_like(x)
cl.enqueue_copy(queue, squared_gpu, output_buf)
queue.finish()

gpu_time_full = time.perf_counter() - t0  # kernel + másolás

# Kernel-only idő az OpenCL event alapján (nanosec → ms)
kernel_ns   = event.profile.end - event.profile.start
kernel_time = kernel_ns / 1e6  # ms

gpu_result = float(np.sum(squared_gpu))

print(f"  Eredmény             = {gpu_result:.6f}")
print(f"  Kernel idő           = {kernel_time:.2f} ms")
print(f"  Teljes pipeline idő  = {gpu_time_full * 1000:.2f} ms  (kernel + host↔device)")


# Részeredmény-ellenőrzés
_banner("Helyesség-ellenőrzés")

cpu_squared = x ** 2
max_elem_diff = float(np.max(np.abs(cpu_squared - squared_gpu)))
print(f"  Max elem-szintű eltérés: {max_elem_diff:.2e}")
_check_result(cpu_result, gpu_result)


# Gyorsítás összefoglalás
_banner("Teljesítmény összefoglalás")

speedup_full   = cpu_time / gpu_time_full
speedup_kernel = (cpu_time * 1000) / kernel_time  # ms vs ms

print(f"  CPU idő              : {cpu_time * 1000:.2f} ms")
print(f"  GPU kernel idő       : {kernel_time:.2f} ms")
print(f"  GPU teljes pipeline  : {gpu_time_full * 1000:.2f} ms")
print(f"  Gyorsítás (kernel)   : {speedup_kernel:.1f}×")
print(f"  Gyorsítás (pipeline) : {speedup_full:.1f}×")

# Eredmények exportálása a grafikon szkript számára
results = {
    "N"              : N,
    "cpu_time_ms"    : cpu_time * 1000,
    "gpu_kernel_ms"  : kernel_time,
    "gpu_pipeline_ms": gpu_time_full * 1000,
    "cpu_result"     : float(cpu_result),
    "gpu_result"     : float(gpu_result),
    "speedup_kernel" : speedup_kernel,
    "speedup_pipeline": speedup_full,
}

import json
out_json = os.path.join(os.path.dirname(__file__), "results.json")
with open(out_json, "w") as fh:
    json.dump(results, fh, indent=2)

print(f"\n  Eredmények mentve → {out_json}")
print("\n  Futtasd a grafikonhoz: python plot_results.py\n")