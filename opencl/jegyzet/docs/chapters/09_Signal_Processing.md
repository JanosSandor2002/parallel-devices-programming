# Jelfeldolgozás

**Jel**: Egy időben változó érték, $s(t)$

* $s$: *signal*
* $t$: *time*

Rövidítések:

* DFT: *Discrete Fourier Transform*
* FFT: *Fast Fourier Transform*

## Komplex számok

Euler reláció: $e^{i\varphi} = \cos \varphi + i \cdot \sin \varphi$

Jelöljön $\overline{z}$ egy komplex számot!

* Algebrai alak: $\overline{z} = a + ib$
* Trigonometrikus alak: $\overline{z} = r \cdot (\cos \varphi + i \cdot \sin \varphi)$
* Euler alak: $\overline{z} = r \cdot e^{i\varphi}$

## Fourier transzformáció

* $s(t)$: a felbontandó periodikus jel
* $s_n(t) \approx s(t)$: a periodikus jel közelítése
* $T$: a jel periódusideje (mértékegysége: *s*)
* $\omega = \dfrac{2\pi}{T}$: körfrekvencia (mértékegysége: *Hz*)
* $S_0 \in \mathbb{R}$: középérték
* $S_{k}^{A}, S_{k}^{B} \in \mathbb{R}$: Fourier együtthatók

A felharmónikusokkal való közelítést a következő összefüggés írja le:

$$
s(t) \approx s_n(t) = S_0 + \sum_{k=1}^{n}
\left[
S_{k}^{A} \cos k \omega t + S_{k}^{B} \sin k \omega t
\right].
$$

Spektrum (frekvencia tartományban való leírás):

$$
S(i\omega) =
\mathcal{F}\{s(t)\} = \int_{-\infty}^{+\infty} \! s(t) \cdot e^{-i \omega t} \mathrm{d}t.
$$

Inverz Fourier transzformáció:

$$
s(t) =
\mathcal{F}^{-1}\{S(i\omega)\} =
\dfrac{1}{2\pi} \cdot
\int_{-\infty}^{+\infty} \! S(i\omega) \cdot e^{i \omega t} \mathrm{d}\omega.
$$

### Tulajdonságok

**Linearitás**

$$
\mathcal{F}\{C_1 s_1(t) + C_2 s_2(t)\} =
C_1 \cdot \mathcal{F}\{s_1(t)\} +
C_2 \cdot \mathcal{F}\{s_2(t)\}
$$

$$
\mathcal{F}^{-1}\{C_1 S_1(i\omega) + C_2 S_2(i\omega)\} =
C_1 \mathcal{F}^{-1}\{S_1(i\omega)\} + C_2 \mathcal{F}^{-1}\{S_2(i\omega)\}
$$

**Eltolás**

$$
\mathcal{F}\{s(t - \tau)\} =
e^{-i \omega \tau} \cdot S(i\omega)
$$

**Konvolúció**

$$
y(t) = w(t) * s(t)
$$

$$
\mathcal{F}\{y(t)\} = \mathcal{F}\{w(t)\} \cdot \mathcal{F}\{s(t)\}
$$

$$
Y(i\omega) = W(i\omega) \cdot S(i\omega)
$$

**Deriválás**

$$
\mathcal{F}\{\dot{s}(t)\} = i\omega \cdot S(i\omega)
$$

## Diszkrét Fourier transzformáció

Diszkrét esetben pontos a felbontás.

* $K \in \mathbb{N}$: a mintavételi pontok száma
* $T_s$: egyenközű mintavételezés esetén a mintavételi idő
* $s[k]$: a $k$-adik mintavételezett pont értéke
* $\vartheta = \dfrac{2\pi}{K}$: alapkörfrekvencia

A Fourier együtthatók számítása:

$$
\overline{S}_p =
\dfrac{1}{K} \cdot
\sum_{k=0}^{K-1} s[k] \cdot e^{-ip\vartheta k}.
$$

Ebből kiadódik, hogy a középérték az átlag:

$$
S_0 =
\dfrac{1}{K} \sum_{k=0}^{K-1} s[k].
$$

A Fourier együtthatók alapján a jelet a következőképpen számíthatjuk:

$$
s[k] =
\sum_{p=0}^{K-1} \overline{S}_p e^{ip\vartheta k}.
$$

Forrás:

* [https://hu.wikipedia.org/wiki/Fourier-transzform%C3%A1ci%C3%B3](https://hu.wikipedia.org/wiki/Fourier-transzformáció)
* [http://maxwell.sze.hu/JELEK/Jelek_es_rendszerek.pdf](http://maxwell.sze.hu/JELEK/Jelek_es_rendszerek.pdf)

## Gyors Fourier Transformáció

* *Cooley-Tukey algoritmus*
* Carl Friedrich Gauss (1805), James Cooley, John Tukey (1965)
* [https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
* [https://medium.com/0xcode/fast-fourier-transform-fft-algorithm-implementation-in-python-b592099bdb27](https://medium.com/0xcode/fast-fourier-transform-fft-algorithm-implementation-in-python-b592099bdb27)

## Hangfájlok feldolgozása

* [https://docs.fileformat.com/audio/wav/](https://docs.fileformat.com/audio/wav/)
* [http://truelogic.org/wordpress/2015/09/04/parsing-a-wav-file-in-c/](http://truelogic.org/wordpress/2015/09/04/parsing-a-wav-file-in-c/)

A betöltött hangfájlt egy jelnek, mint amplitúdó sorozatnak tekinthetjük.

* Konvolúció, simítás
* Szűrés, alul- és felüláteresztő szűrő

## Feladatok

**1. FFT algoritmus**

* Implementáljuk az FFT algoritmust C-ben!
* Ellenőrízzük a számításokat 8 elemű példákon!

**2. Harmonikus függvények vizsgálata**

* Vizsgáljuk meg a DFFT algoritmus működését $\sin$ és $\cos$ függvények mintavételezésére!
* Adjunk meg előre olyan frekvenciákat, amelyeket eredményként várunk majd!
* Vizsgáljuk meg a becslés pontosságát a mintavételezés függvényében!

**3. Hangfájlok kezelése**

* Készítsünk egy programot, amelyik adott frekvenciájú, szinuszos jelet ír ki egy `wav` fájlba!
* Generáljunk olyan jeleket, amelyek egyidejűleg több frekvenciát is tartalmaznak!

**4. Szűrés**

* Implementáljunk (például egy) átlagoló szűrőt!
* Valósítsuk meg OpenCL segítségével!
* Mérésekkel vizsgáljuk meg, hogy a párhuzamos implementáció hatékonyabb, mint a szekvenciális!

