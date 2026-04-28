# GPU OpenCL Benchmark – teljes összehasonlító mérés (3 gép, 2 OS, AMD + NVIDIA)

## Projekt cél

Ez a projekt OpenCL alapú GPU gyorsítást vizsgál nagy méretű vektorokon:

- N = 10,000,000 float32 elem
- művelet: x² + redukció (tree reduction)
- CPU (NumPy) vs GPU (OpenCL kernel)
- kernel idő + teljes pipeline idő
- több hardver + több operációs rendszer összehasonlítása

## Hardverek

### 1. Régi PC (Windows 10)

- CPU: Intel Core i5-6500
  - 4 mag / 4 szál
  - 3.2 GHz
- RAM: 8 GB DDR3
- GPU: NVIDIA GeForce GT 1030
  - Compute Units: 3
  - VRAM: 2 GB
- OpenCL: NVIDIA CUDA stack

### 2. Új PC (Windows 10)

- CPU: AMD Ryzen 5 PRO 2400G
  - 4 mag / 8 szál
  - 3.6 GHz
- RAM: 16 GB DDR4
- GPU: AMD Radeon Vega iGPU (gfx902)
  - Compute Units: 11
  - Shared memória: 6.4 GB
- OpenCL: AMD APP / ROCm stack

### 3. Laptop (Windows 11 + NixOS dual benchmark)

- CPU: AMD Ryzen 5 7533HS
  - 6 mag / 12 szál
  - akár 4.45 GHz boost
- RAM: 16 GB DDR5
- GPU: AMD Radeon 660M (RDNA2 iGPU)
- OpenCL:
  - Windows 11: AMD Adrenalin driver
  - Linux (NixOS): Mesa + ROCm / LLVM OpenCL

## Benchmark feladat

- Input: 10,000,000 float32 szám
- memória: ~38.1 MB
- művelet:
  - x²
  - hierarchical reduction (GPU tree reduction)

Reduction lánc:
10,000,000 → 39,063 → 153 → 1

## 1. Régi PC (Windows 10, GT 1030)

CPU:

- ~21 – 25 ms

GPU kernel:

- ~4.7 – 6.2 ms

GPU pipeline:

- ~5.2 – 7.0 ms

Gyorsítás:

- Kernel: 4.0× – 4.8×
- Pipeline: 3.4× – 4.4×

Jellemzők:

- stabil, kis szórás
- alacsony compute unit szám
- kis, de konzisztens gyorsítás

## 2. Új PC (Windows 10, Vega iGPU gfx902)

CPU:

- ~32 – 42 ms

GPU kernel:

- ~2.4 – 6.6 ms

GPU pipeline:

- ~3.1 – 19.2 ms

Gyorsítás:

- Kernel: 8× – 13×
- Pipeline: 2× – 10×

Jellemzők:

- magas CU szám (11)
- nagy elméleti teljesítmény
- pipeline instabilitás (driver + memória overhead)

## 3. Laptop (AMD Ryzen 5 7533HS + Radeon 660M)

### Windows 11 (AMD Adrenalin)

CPU:

- 25.55 ms

GPU kernel:

- 2.37ms

GPU pipeline:

- 3.92 ms

Gyorsítás:

- Kernel: 10.8×
- Pipeline: 6.5×

Jellemzők:

- stabil driver stack
- alacsony szórás
- kis-közepes GPU gyorsítás

### Linux (NixOS + Mesa / ROCm)

CPU:

- 8 – 16 ms (tipikus)

GPU kernel:

- 6.2 – 15.0 ms

GPU pipeline:

- 10.9 – 14.9 ms (tipikus)

Gyorsítás:

- Kernel: 0.6× – 5.4×
- Pipeline: 1.0× – 4.8×

Jellemzők:

- nagy szórás
- Mesa OpenCL korlátok
- compute unit instabilitás
- gyakori fallback viselkedés

## Összehasonlítás (összes gép)

### GPU kernel gyorsítás

- GT 1030: 4× – 5×
- Vega gfx902: 8× – 13×
- Radeon 660M:
  - Windows: 6.5× - 10.8x
  - Linux: 0.6× – 5.4x

### Pipeline gyorsítás

- GT 1030: 3.4× – 4.4×
- Vega gfx902: 2× – 10×
- Radeon 660M:
  - Windows: 6.5×
  - Linux: 1.0× – 4.8×

### Stabilitás

| Rendszer         | Stabilitás |
| ---------------- | ---------- |
| NVIDIA (GT 1030) | magas      |
| AMD Windows      | közepes    |
| AMD Linux        | alacsony   |

## Driver hatás

### NVIDIA (GT 1030)

- stabil OpenCL runtime
- kis CU szám → kis variancia
- kis, de megbízható gyorsítás

### AMD Windows (Adrenalin)

- optimalizált OpenCL stack
- jó kernel indítás
- stabil memory összevonás (thread0 = x[0], thread1 = x[1])

### AMD Linux (Mesa / ROCm)

- LLVM JIT overhead
- compute unit kihasználás változó
- kernel cache érzékenység
- pipeline overhead magas

## GPU architektúra hatás

- RDNA2 / Vega:
  - nagy párhuzamosság
  - de memória + driver limit
- NVIDIA GT 1030:
  - kisebb, de determinisztikus viselkedés

## Fő tanulság

1. A GPU teljesítmény nem csak hardverfüggő
2. A driver stack kritikus:
   - Windows AMD >> Linux AMD stabilitás
3. iGPU rendszereknél a memória a fő limit
4. Pipeline overhead sokszor nagyobb hatású, mint a compute

## Összegzés

- NVIDIA: stabil, kisebb gyorsítás
- AMD Windows: legjobb egyensúly iGPU-n
- AMD Linux: legnagyobb ingadozás
- új Ryzen laptop:
  - modern CPU
  - erős iGPU
  - de driver függő GPU kihasználás
