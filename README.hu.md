[🇬🇧 English](README.md) | 🇭🇺 Magyar

# parallel-devices-programming

Repository a Miskolci Egyetem **Párhuzamos eszközök programozása** tantárgyához.

## A repóról

Ez a repó a félév során elkészített **gyakorlati feladatokat** tartalmazza, valamint a **beadandót** — egy OpenCL alapú GPU benchmark projektet, amely CPU és GPU teljesítményt hasonlít össze több gépen és operációs rendszeren keresztül.

## Tartalom

- **Gyakorlati feladatok** — a félév során elkészített kisebb, kézzelfogható feladatok a párhuzamos/GPU programozási koncepciók gyakorlására.
- **Beadandó: GPU OpenCL Benchmark (négyzetösszeg)** — egy részletesebb projekt, lásd lentebb.

---

## Beadandó: Vektorok négyzetösszege — CPU vs GPU (OpenCL) Benchmark

### Feladat leírása

- Egy nagyméretű tömb (`N = 10 000 000`) `float32` lebegőpontos számokkal.
- Minden elem négyzetre emelése, majd az eredmények összegzése (**sum of squares**).
- A számítás **OpenCL kernelben** történik, összehasonlítva egy **CPU (NumPy) implementációval**.
- A futási idő mérése mindkét módszerrel, a gyorsulás láthatóvá tételéhez.
- Az eredmények exportálásra kerülnek, és összehasonlító **grafikonok** készülnek belőlük.

### Fájlok

| Fájl                | Leírás                                              |
| ------------------- | --------------------------------------------------- |
| `sum_of_squares.cl` | OpenCL kernel — elem-szintű négyzetre emelés GPU-n  |
| `main.py`           | Fő benchmark szkript — CPU + GPU mérés, JSON export |
| `results.py`        | Grafikon szkript — 6 panel, dark theme              |
| `results.json`      | Automatikusan generálódik `main.py` futtatásakor    |
| `plots.png`         | Automatikusan generálódik `results.py` futtatásakor |

### Telepítés és futtatás

**Függőségek:**

```powershell
py -m pip install pyopencl numpy matplotlib
```

**Futtatás:**

```powershell
# 1. Benchmark – méri a CPU és GPU időt, elmenti az eredményeket
py main.py

# 2. Grafikonok generálása az eredményekből
py results.py
```

> A `main.py`-t mindig előbb kell futtatni, mert létrehozza a `results.json` fájlt, amelyet a `results.py` olvas be.

### Mit mér a program?

| Mérőszám                  | Leírás                                               |
| ------------------------- | ---------------------------------------------------- |
| **CPU idő**               | NumPy `np.sum(x**2)` végrehajtási ideje              |
| **GPU kernel idő**        | Csak az OpenCL kernel futási ideje (event profiling) |
| **GPU pipeline idő**      | Kernel + host↔device memóriaátvitel együtt           |
| **Gyorsítás (×)**         | CPU idő / GPU idő — mindkét GPU mérőszámra           |
| **Numerikus pontosság**   | CPU és GPU eredmény relatív eltérése                 |
| **Effektív sávszélesség** | `2 × N × 4 byte / kernel_idő` (GB/s)                 |

### Algoritmus — Tree Reduction

A GPU-s számítás **hierarchikus (tree) reduction** módszert használ:

1. **Elem-szintű feldolgozás** — minden work-item kiszámolja: `x_i²`
2. **Work-group szintű összeadás** — lokális memória (`__local`) használatával, stride-alapú bináris összeadás
3. **Többlépcsős redukció** — a partial eredmények újra GPU kernelbe kerülnek, amíg 1 érték nem marad

Reduction lánc: `10 000 000 → 39 063 → 153 → 1` — ez O(log N) mélységű hierarchiát jelent.

Az implementáció két kernelt használ: `sum_of_squares` (kezdeti négyzetre emelés + első reduction pass) és `reduce` (további hierarchikus összeadás), work-groupönként 256 elemmel és barrier szinkronizációval.

### Mérési eredmények (egygépes alapmérés)

**Futási idők:**

| Módszer             | Idő      |
| ------------------- | -------- |
| CPU (NumPy)         | 25.55 ms |
| GPU kernel          | 2.37 ms  |
| GPU teljes pipeline | 3.92 ms  |

**Gyorsítás:**

| Metrika         | Gyorsítás |
| --------------- | --------- |
| Kernel-only     | **10.8×** |
| Teljes pipeline | **6.5×**  |

**Numerikus pontosság:** CPU eredmény `3332497.25`, GPU eredmény `3332497.5` — relatív eltérés `7.5 × 10⁻⁸` (helyes egyezés).

**Pipeline bontás:** ~2.37 ms kernel execution, ~1.5 ms host + memória overhead — a teljes futás nem csak compute-ot, hanem memória mozgatást és launch overheadet is tartalmaz.

---

## Kiterjesztett benchmark: több gép, több OS összehasonlítás

Ugyanaz a benchmark (N = 10 000 000, x² + tree reduction) lefutott **3 gépen és 2 operációs rendszeren** (AMD + NVIDIA GPU-kkal), hogy a hardver és driver hatásokat is össze lehessen hasonlítani.

### Tesztelt hardverek

**1. Régi PC (Windows 10)**

- CPU: Intel Core i5-6500 (4 mag/4 szál, 3.2 GHz), 8 GB DDR3
- GPU: NVIDIA GeForce GT 1030 (3 Compute Unit, 2 GB VRAM), NVIDIA CUDA OpenCL stack

**2. Új PC (Windows 10)**

- CPU: AMD Ryzen 5 PRO 2400G (4 mag/8 szál, 3.6 GHz), 16 GB DDR4
- GPU: AMD Radeon Vega iGPU / gfx902 (11 Compute Unit, 6.4 GB megosztott memória), AMD APP/ROCm stack

**3. Laptop (Windows 11 + NixOS dual benchmark)**

- CPU: AMD Ryzen 5 7533HS (6 mag/12 szál, akár 4.45 GHz boost), 16 GB DDR5
- GPU: AMD Radeon 660M (RDNA2 iGPU)
- OpenCL: AMD Adrenalin driver (Windows) / Mesa + ROCm/LLVM OpenCL (NixOS)

### Eredmények összefoglalva

| Rendszer            | CPU idő   | GPU kernel  | GPU pipeline | Kernel gyorsítás | Pipeline gyorsítás |
| ------------------- | --------- | ----------- | ------------ | ---------------- | ------------------ |
| GT 1030 (Win10)     | ~21–25 ms | ~4.7–6.2 ms | ~5.2–7.0 ms  | 4.0×–4.8×        | 3.4×–4.4×          |
| Vega gfx902 (Win10) | ~32–42 ms | ~2.4–6.6 ms | ~3.1–19.2 ms | 8×–13×           | 2×–10×             |
| Radeon 660M (Win11) | 25.55 ms  | 2.37 ms     | 3.92 ms      | 10.8×            | 6.5×               |
| Radeon 660M (NixOS) | 8–16 ms   | 6.2–15.0 ms | 10.9–14.9 ms | 0.6×–5.4×        | 1.0×–4.8×          |

### Stabilitás rendszerenként

| Rendszer         | Stabilitás |
| ---------------- | ---------- |
| NVIDIA (GT 1030) | Magas      |
| AMD Windows      | Közepes    |
| AMD Linux        | Alacsony   |

### Fő tanulságok

1. A GPU teljesítmény nem kizárólag hardverfüggő — a driver stack is jelentősen számít.
2. A Windows-os AMD driver (Adrenalin) érezhetően stabilabb, mint a Linux-os AMD (Mesa/ROCm).
3. iGPU-s rendszereknél a megosztott memória sávszélessége gyakran a fő korlát.
4. A pipeline overhead (memóriaátvitel, kernel indítás) sokszor nagyobb hatású, mint a nyers compute idő.
5. Az NVIDIA GT 1030 kisebb, de nagyon konzisztens/determinisztikus gyorsítást mutat; az AMD Vega/RDNA2 iGPU-k nagyobb csúcsgyorsítást nyújtanak, de nagyobb szórással, főleg Linux alatt.

## Technológiai stack

- **Python** (benchmark orchestráció, NumPy a CPU alapmérésekhez, Matplotlib a grafikonokhoz)
- **OpenCL** (PyOpenCL) a GPU kernelekhez
- **C** (OpenCL kernel kód)

## A repó célja

Ez a repó egyrészt gyakorlóterep a tantárgy során tanult párhuzamos/GPU programozási koncepciókhoz, másrészt a beadandó dokumentációja — bemutatva az OpenCL tree reduction működését, a lokális memória optimalizációt, és a platformfüggetlen CPU vs. GPU teljesítmény-összehasonlítást.

## Licenc

Egyetemi/oktatási célú projekt — jelenleg nincs újrafelhasználásra licencelve.
