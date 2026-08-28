🇬🇧 English | [🇭🇺 Magyar](README.hu.md)

# parallel-devices-programming

Repository for the **Parallel Devices Programming** course at the University of Miskolc (Miskolci Egyetem).

## About

This repo contains practical exercises completed throughout the course, along with the **final assignment (beadandó)** — an OpenCL-based GPU benchmark project comparing CPU and GPU performance across multiple machines and operating systems.

## Contents

- **Practical exercises** — smaller hands-on tasks completed during the semester to practice parallel/GPU programming concepts.
- **Assignment: GPU OpenCL Benchmark (sum of squares)** — a more in-depth project, detailed below.

---

## Assignment: Sum of Squares — CPU vs GPU (OpenCL) Benchmark

### Task Description

- A large array (`N = 10,000,000`) of `float32` numbers is generated.
- Each element is squared, and the results are summed (**sum of squares**).
- The computation is performed in an **OpenCL kernel**, and compared against a **CPU (NumPy) implementation**.
- Execution time is measured for both approaches to visualize the speedup.
- Results are exported and used to generate comparison **charts**.

### Files

| File                | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `sum_of_squares.cl` | OpenCL kernel — element-wise squaring on the GPU      |
| `main.py`           | Main benchmark script — CPU + GPU timing, JSON export |
| `results.py`        | Chart script — 6-panel comparison, dark theme         |
| `results.json`      | Auto-generated when running `main.py`                 |
| `plots.png`         | Auto-generated when running `results.py`              |

### Installation & Running

**Dependencies:**

```powershell
py -m pip install pyopencl numpy matplotlib
```

**Running:**

```powershell
# 1. Benchmark – measures CPU and GPU time, saves results
py main.py

# 2. Generate charts from the results
py results.py
```

> `main.py` must always be run first, as it generates the `results.json` file that `results.py` reads.

### What the Program Measures

| Metric                  | Description                                         |
| ----------------------- | --------------------------------------------------- |
| **CPU time**            | Execution time of NumPy's `np.sum(x**2)`            |
| **GPU kernel time**     | OpenCL kernel execution time only (event profiling) |
| **GPU pipeline time**   | Kernel + host↔device memory transfer combined       |
| **Speedup (×)**         | CPU time / GPU time, for both GPU metrics           |
| **Numerical accuracy**  | Relative difference between CPU and GPU results     |
| **Effective bandwidth** | `2 × N × 4 bytes / kernel_time` (GB/s)              |

### Algorithm — Tree Reduction

The GPU computation uses a **hierarchical (tree) reduction**:

1. **Element-level processing** — each work-item computes `x_i²`
2. **Work-group level summation** — using local memory (`__local`), stride-based binary addition
3. **Multi-pass reduction** — partial results are fed back into the GPU kernel until a single value remains

Reduction chain: `10,000,000 → 39,063 → 153 → 1` — an O(log N) depth hierarchy.

The implementation uses two kernels: `sum_of_squares` (initial squaring + first reduction pass) and `reduce` (further hierarchical summation), with 256 elements per work-group and barrier synchronization.

### Benchmark Results (single-machine baseline)

**Execution times:**

| Method            | Time     |
| ----------------- | -------- |
| CPU (NumPy)       | 25.55 ms |
| GPU kernel        | 2.37 ms  |
| GPU full pipeline | 3.92 ms  |

**Speedup:**

| Metric        | Speedup   |
| ------------- | --------- |
| Kernel-only   | **10.8×** |
| Full pipeline | **6.5×**  |

**Numerical accuracy:** CPU result `3332497.25` vs. GPU result `3332497.5` — relative difference of `7.5 × 10⁻⁸` (correct match).

**Pipeline breakdown:** ~2.37 ms kernel execution, ~1.5 ms host + memory overhead — the full run includes not just compute time but also memory transfer and launch overhead.

---

## Extended Benchmark: Multi-Machine, Multi-OS Comparison

The same benchmark (N = 10,000,000, x² + tree reduction) was run across **3 machines and 2 operating systems** (AMD + NVIDIA GPUs) to compare hardware and driver effects.

### Hardware Tested

**1. Old PC (Windows 10)**

- CPU: Intel Core i5-6500 (4 cores/4 threads, 3.2 GHz), 8 GB DDR3
- GPU: NVIDIA GeForce GT 1030 (3 Compute Units, 2 GB VRAM), NVIDIA CUDA OpenCL stack

**2. New PC (Windows 10)**

- CPU: AMD Ryzen 5 PRO 2400G (4 cores/8 threads, 3.6 GHz), 16 GB DDR4
- GPU: AMD Radeon Vega iGPU / gfx902 (11 Compute Units, 6.4 GB shared memory), AMD APP/ROCm stack

**3. Laptop (Windows 11 + NixOS dual benchmark)**

- CPU: AMD Ryzen 5 7533HS (6 cores/12 threads, up to 4.45 GHz boost), 16 GB DDR5
- GPU: AMD Radeon 660M (RDNA2 iGPU)
- OpenCL: AMD Adrenalin driver (Windows) / Mesa + ROCm/LLVM OpenCL (NixOS)

### Results Summary

| System              | CPU time  | GPU kernel  | GPU pipeline | Kernel speedup | Pipeline speedup |
| ------------------- | --------- | ----------- | ------------ | -------------- | ---------------- |
| GT 1030 (Win10)     | ~21–25 ms | ~4.7–6.2 ms | ~5.2–7.0 ms  | 4.0×–4.8×      | 3.4×–4.4×        |
| Vega gfx902 (Win10) | ~32–42 ms | ~2.4–6.6 ms | ~3.1–19.2 ms | 8×–13×         | 2×–10×           |
| Radeon 660M (Win11) | 25.55 ms  | 2.37 ms     | 3.92 ms      | 10.8×          | 6.5×             |
| Radeon 660M (NixOS) | 8–16 ms   | 6.2–15.0 ms | 10.9–14.9 ms | 0.6×–5.4×      | 1.0×–4.8×        |

### Stability by System

| System           | Stability |
| ---------------- | --------- |
| NVIDIA (GT 1030) | High      |
| AMD Windows      | Medium    |
| AMD Linux        | Low       |

### Key Takeaways

1. GPU performance is not purely hardware-dependent — the driver stack matters significantly.
2. Windows AMD driver (Adrenalin) is notably more stable than Linux AMD (Mesa/ROCm).
3. On iGPU systems, shared memory bandwidth is often the main bottleneck.
4. Pipeline overhead (memory transfer, kernel launch) frequently outweighs raw compute time.
5. NVIDIA's GT 1030 shows smaller but highly consistent/deterministic speedups; AMD's Vega/RDNA2 iGPUs offer higher peak speedup but with more variance, especially under Linux.

## Tech Stack

- **Python** (benchmark orchestration, NumPy for CPU baseline, Matplotlib for charts)
- **OpenCL** (PyOpenCL) for GPU kernels
- **C** (OpenCL kernel code)

## Goals of This Repository

This repo serves as both a practice space for parallel/GPU programming concepts covered in the course, and documentation of the final assignment — demonstrating OpenCL tree reduction, local memory optimization, and a cross-platform CPU vs. GPU performance comparison.

## License

Academic coursework project — not currently licensed for reuse.
