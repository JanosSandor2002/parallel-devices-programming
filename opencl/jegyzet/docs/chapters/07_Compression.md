# Tömörítési algoritmusok

!!! task "Szempontok"

    Milyen szempontokat érdemes figyelembe venni egy tömörítési algoritmus tervezése/használata esetén?

<!--
* Adatok visszaállíthatósága: veszteséges vagy veszteségmentes
* Tömörítési arány
* Ki- és betömörítési idő
* Kell-e ismerni a tömörítéshez a teljes fájlt?
* Általános, vagy specializált tömörítési eljárásról van-e szó?
* Paraméterezhetőség
-->

* Veszteségmentes tömörítés
* Futáshossz kódolás
* Entrópia kódolás
* [https://en.wikipedia.org/wiki/Lossless_compression](https://en.wikipedia.org/wiki/Lossless_compression)

!!! warning "Hibajavítás"

    A tömörítési eljárások érzékenyek a tömörített adatokben keletkezett hibákra.

!!! warning "Licenszek"

    Bizonyos tömörítési eljárásokat licensz véd.

## Futáshossz kódolás

* RLE: *Run Length Encoding*
* [https://en.wikipedia.org/wiki/Run-length_encoding](https://en.wikipedia.org/wiki/Run-length_encoding)

### Lempel-Ziv algoritmus

* Abraham Lempel, Jacob Ziv, 1977
* LZ77, LZ78
* [https://en.wikipedia.org/wiki/LZ77_and_LZ78](https://en.wikipedia.org/wiki/LZ77_and_LZ78)

### LZW

* Terry Welch, 1984
* Az LZ78 fejlesztett változata
* [https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch)

### LZMA

* Igor Pavlov, 1998
* Szótár alapú tömörítő eljárás
* 7-Zip tömörítő, `.7z`
* Bit alapú eljárás
* [https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm)

### LZSS

* James A. Storer, Thomas Szymanski, 1982
* A hivatkozott szövegrészeknél mindig rövidebb hivatkozásokat használ.
* [https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Storer%E2%80%93Szymanski](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Storer%E2%80%93Szymanski)

## Huffman kódolás

* A gyakrabban előforduló blokkokat rövidebb, a ritkábban előfordulóakat hosszabb kódszavakkal kódoljuk.
* Entrópia: [https://en.wikipedia.org/wiki/Entropy_(information_theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
* [https://en.wikipedia.org/wiki/Huffman_coding](https://en.wikipedia.org/wiki/Huffman_coding)

## Kiegészítő algoritmusok

### Move-To-Front transzformáció

* [https://en.wikipedia.org/wiki/Move-to-front_transform](https://en.wikipedia.org/wiki/Move-to-front_transform)

### Burrows-Wheeler transzformáció

* Michael Burrows, David Wheeler, 1994
* [https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform](https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform)

Az algoritmus lépései:

* Soroljuk fel a bemeneti szöveg összes forgatását!
* Rendezzük őket növekvő sorrendbe az első karakterük szerint!
* A kimenetet az utolsó karakterek adják.

## DEFLATE

* A korábbi algoritmusok egy hatékony kombinációja.
* [https://en.wikipedia.org/wiki/Deflate](https://en.wikipedia.org/wiki/Deflate)

ZIP implementációk

* [https://en.wikipedia.org/wiki/Gzip](https://en.wikipedia.org/wiki/Gzip)
* [https://en.wikipedia.org/wiki/Bzip2](https://en.wikipedia.org/wiki/Bzip2)

## Feladatok

**1. Leghosszabb egyezés keresése**

* Implementáljunk egy algoritmust, amelyik bemenetként kap egy hosszabb szöveget (*window*) és egy rövidebbet (*input*), majd megkeresi a leghosszabb prefix indexét és annak hosszát!
* Párhuzamosítsuk a számítást OpenCL segítségével!
* Készítsük el ennek segítségével az LZ77 algoritmus egy egyszerűbb változatát!

**2. Futáshossz kódolás implementációk**

* Tekintsünk át néhány implementációt!
* [https://rosettacode.org/wiki/Run-length_encoding](https://rosettacode.org/wiki/Run-length_encoding)

**3. Gyakoriságok számítása**

* Készítsünk egy programot OpenCL segítségével, amely egy bináris tömbben meghatározza az előforduló byte értékek gyakoriságát!

**4. Huffman kódolás**

* A gyakoriságok alapján készítsünk Huffman kódoláshoz kódfát!
* Valósítsuk meg a kódolást és dekódolást!

**5. Blokkméret optimalizáció**

* OpenCL és mérések segítségével próbáljunk meg becslést adni Huffman kódolás esetén az optimális blokkméretre!
* Magát a mérést párhuzamosítsuk!
* Egyszerűbb esetben byte, kompikáltabb változatban bit alapon kódoljunk!

**6. Entrópia csökkentése**

* Vizsgáljuk meg mérések segítségével, hogy a *Move-To-Front* algoritmus hogyan csökkenti az entrópiát!
* Összegezzük, ábrázoljuk a kapott eredményeket!

