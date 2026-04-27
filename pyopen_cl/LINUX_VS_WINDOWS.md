# AMD iGPU OpenCL Benchmark – Linux vs Windows (részletes mérési adatokkal)

## Linux rendszer

### Rendszer

- NixOS (Linux kernel)
- AMD integrált GPU (RDNA2 / gfx1035)
- CPU: AMD Ryzen APU (integrált memóriaarchitektúra miatt shared RAM/VRAM)

### GPU / driver stack

- amdgpu kernel driver
- Mesa graphics stack (OpenCL implementáció)
- OpenCL: rocmPackages.clr.icd (ROCm userspace Mesa alatt)
- LLVM alapú OpenCL kernel JIT fordítás
- PyOpenCL Python wrapper

### Linux környezet jellemzői

- dinamikus GPU power management (frekvencia scaling futás közben)
- ROCm / Mesa OpenCL réteg (nem teljes Adrenalin stack)
- változó kernel scheduling (CPU–GPU verseny iGPU memória miatt)
- shared memory bandwidth limit (nincs dedikált VRAM)
- első futásnál OpenCL kernel compile overhead

### Linux benchmark részletes eredmények

#### Tesztparaméterek

- Input méret: 10,000,000 float32 elem
- Memóriahasználat: ~38.1 MB
- Művelet: parallel reduction OpenCL kernel
- GPU pipeline: 2-pass reduction

### CPU számítás (NumPy)

- Minimum: 8.0 ms
- Maximum: 19.0 ms
- Tipikus: 9.0 ms – 16.0 ms
- Átlag: stabil, kis szórás
- Függ:
  - CPU boost
  - háttérterhelés
  - Python warmup

### GPU számítás (OpenCL)

Kernel idő:

- Minimum: 6.2 ms
- Maximum: 15.0 ms
- Tipikus: 10.5 ms – 13.5 ms

Teljes pipeline idő:

- Minimum: 6.6 ms
- Maximum: 15.0 ms
- Tipikus: 10.9 ms – 14.9 ms

### GPU gyorsítás (Linux)

- Minimum: 0.6×
- Maximum: 2.4×
- Tipikus: 1.0× – 1.5×

### Megfigyelések

- jelentős futásról futásra szórás (±30–60%)
- kernel compile cache erősen befolyásolja az első futást
- Mesa/OpenCL optimalizáció korlátozott
- compute unit detektálás: gyakran 3 CU aktív
- memória sávszélesség limitáló tényező

## Windows rendszer

### Rendszer

- Windows 11
- AMD integrált GPU (azonos hardver: RDNA2 / gfx1035)
- CPU: AMD Ryzen APU

### GPU / driver stack

- AMD Adrenalin driver (teljes OpenCL implementáció)
- optimalizált GPU runtime
- hardverközeli scheduling
- fejlettebb memory coalescing
- gyorsabb kernel dispatch

### Windows környezet jellemzői

- stabil GPU power management
- optimalizált OpenCL futtatás
- kisebb JIT overhead
- gyorsabb kernel startup

### Windows benchmark részletes eredmények

### CPU számítás

- Minimum: 8.0 ms
- Maximum: 19.0 ms
- Tipikus: 9.0 ms – 15.0 ms

### GPU számítás

Kernel idő:

- Minimum: 5.5 ms
- Maximum: 10.5 ms
- Tipikus: 6.0 ms – 9.0 ms

Teljes pipeline:

- Minimum: 6.0 ms
- Maximum: 11.0 ms
- Tipikus: 6.5 ms – 9.5 ms

### GPU gyorsítás (Windows)

- Minimum: 1.2×
- Maximum: 1.7×
- Tipikus: 1.3× – 1.6×

## Összehasonlítás

### Teljesítmény

- Windows: 1.3× – 1.7× GPU gyorsítás
- Linux: 0.6× – 2.4× GPU gyorsítás

### Stabilitás

- Windows: stabil, kis szórás (~10–15%)
- Linux: erős ingadozás (~30–60%)

### Driver viselkedés

- Windows: AMD Adrenalin (optimalizált zárt stack)
- Linux: Mesa + ROCm + LLVM JIT stack

### GPU kihasználtság

- Windows: konzisztens magas kihasználás
- Linux: változó CU kihasználás, néha fallback viselkedés

## Következtetés

A különbség nem a hardverből, hanem a driver stack-ből adódik.

- Windows OpenCL stack jobban optimalizált AMD iGPU-ra
- Linux OpenCL stack rugalmasabb, de kevésbé konzisztens
- ugyanaz a kód eltérő GPU viselkedést és teljesítményt mutat
