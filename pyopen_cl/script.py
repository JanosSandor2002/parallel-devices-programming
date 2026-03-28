import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

# -------------------------------
# Beállítások
N = 10_000_000
np.random.seed(42)

# Véletlenszámok
x = np.random.rand(N).astype(np.float32)

# -------------------------------
# CPU számítás (NumPy)
start_cpu = time.time()
cpu_result = np.sum(x ** 2)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print(f"CPU sum of squares: {cpu_result:.6f}, time: {cpu_time:.4f}s")

# -------------------------------
# OpenCL setup
platforms = cl.get_platforms()
gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=gpu_devices)
queue = cl.CommandQueue(ctx)

# Kernel betöltése
with open("sum_of_squares.cl", "r") as f:
    kernel_source = f.read()

program = cl.Program(ctx, kernel_source).build()

# Memóriák
mf = cl.mem_flags
input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, x.nbytes)  # ugyanakkora méret, minden elemhez

# -------------------------------
# GPU számítás
start_gpu = time.time()
program.sum_of_squares(queue, x.shape, None, input_buf, output_buf)

# Eredmény visszaolvasása
result_gpu = np.empty_like(x)
cl.enqueue_copy(queue, result_gpu, output_buf)
queue.finish()

gpu_result = np.sum(result_gpu)
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"GPU sum of squares: {gpu_result:.6f}, time: {gpu_time:.4f}s")

# -------------------------------
# Gyorsítás
speedup = cpu_time / gpu_time
print(f"Speedup (CPU/GPU): {speedup:.2f}x")

# -------------------------------
# Grafikon
plt.bar(["CPU", "GPU"], [cpu_time, gpu_time], color=["orange", "blue"])
plt.ylabel("Time (s)")
plt.title("Sum of squares: CPU vs GPU")
plt.show()