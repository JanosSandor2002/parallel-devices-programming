# Feladat: Vektorok négyzetösszege (sum of squares)

## Leírás

- Legyen egy nagyméretű tömb (`N = 10_000_000`) lebegőpontos számokkal.
- Számítsa ki minden elem négyzetét, majd összegezze az eredményt (**sum of squares**).
- Az számítás történjen **OpenCL kernelben**, majd **összehasonlításképp CPU-n NumPy-vel**.
- Mérje az időt mindkét módszerrel a gyorsítás láthatóságához.
- A mérési eredményekből **grafikonok készülnek**, amelyek összehasonlítják a GPU és CPU teljesítményét.

## Tesztelhető kritériumok

- `N` véletlenszerű szám (`float32`), például:
  ```python
  np.random.rand(N).astype(np.float32)
  ```
