# Képfeldolgozási eljárások

## Geometriai transzformációk

Alapvető transzformációk:

* Eltolás
* Skálázás
* Forgatás

!!! task "További transzformációk"

    Milyen további transzformációk vannak?

* A mintavételezés miatt mindegyik művelet torzítással, információveszteséggel járhat.
* A transzformáció elvégzése tulajdonképpen újra mintavételezést jelent.
* Az eredeti képhez egy folytonos intenzitásfelületet kell számolni.
* A probléma a nagyításra, mint a felbontás növelésére vezethető vissza.
* A bilineáris interpoláció erre egy egyszerű lehetőséget ad.

!!! task "Interpoláció"

    Milyen további transzformációk jöhetnek még szóba? 

!!! note "Super resolution"

    A *super resolution* is erre ad egy érdekes megoldást.

Forrás:

* [https://en.wikipedia.org/wiki/Geometric_transformation](https://en.wikipedia.org/wiki/Geometric_transformation)
* [https://en.wikipedia.org/wiki/Bilinear_interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation)
* [https://www.atlantis-press.com/article/4822.pdf](https://www.atlantis-press.com/article/4822.pdf)
* [https://www.cambridgeincolour.com/tutorials/image-interpolation.htm](https://www.cambridgeincolour.com/tutorials/image-interpolation.htm)
* [https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms](https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms)

## Hisztogramok számítása

* Jól párhuzamosítható, mivel a lokális hisztogramok összeadhatók.

## Konvolúciós szűrők

* Átlagoló szűrő
* Gauss szűrő
* Medián szűrő

## Kép objektumok kezelése

Képeket bufferekben is lehet tárolni, de célszerűbb az OpenCL kép objektumait használni.

* Ezek a textúra memóriában kerülnek tárolásra, így gyorsabb elérést tesznek lehetővé.
* Az indexelés természetesebben adódik.
* A pixel formátumot maga az OpenCL adja. (A kép betöltését követően azzal már nem kell foglalkozni.)
* 2D és 3D képeket is tudunk a segítségével kezelni.

```c
cl_mem clCreateImage(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format *image_format,
    const cl_image_desc *image_desc,
    void *host_ptr,
    cl_int *errcode_ret
)
```

* [clCreateImage](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateImage.html)

Az adatok írása és olvasása a bufferekhez hasonlóan működik:

* [clEnqueueWriteImage](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueWriteImage.html)
* [clEnqueueReadImage](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueReadImage.html)

Kernelben használható függvények:

* [imageFunctions](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/imageFunctions.html)

## Az OpenCV függvénykönyvtár

* Képfeldolgozáshoz, gépi látáshoz kötődő alkalmazások számára készített függvénykönyvtár.
* [https://opencv.org/](https://opencv.org/)

Képfeldolgozáshoz kapcsolódó függvények:

* [https://docs.opencv.org/4.x/d7/dbd/group__imgproc.html](https://docs.opencv.org/4.x/d7/dbd/group__imgproc.html)

## Kérdések

* Milyen kapcsolat van a *Képfeldolgozás*, *Gépi látás* és *Gépi tanulás* területek között?
* Lehet-e olyan alkalmazást készíteni, amelyik egyidejűleg használ *OpenCL*-t, *OpenCV*-t és *OpenGL*-t?

## Feladatok

**1. Képadatok kezelése**

* Hozzunk létre egy RGB (vagy RGBA) pixel formátumú, $256 \times 256$ méretű képtömböt!
* Állítsuk be, hogy a vörös komponens értéke soronként növekedve $[0, 255]$ intervallumon változzon. Hasonlóképpen adjuk meg az oszlopokra is.
* Küldjük át az adatokat a videókártyára, ahol egy kernel segítségével invertáljuk a színeket!
* Másoljuk vissza a kép objektumot a host programba, és ellenőrízzük a kapott eredményt!

**2. Transzformációs mátrix**

* Készítsünk egy programot, amely tetszőleges $3 \times 3$ méretű transzformációs mátrix segítségével képes képeket transzformálni.
* A mintavételezéshez használjuk a legközelebbi képpontnak a színét!
* Figyeljünk arra, hogy a mintavételezési pont mindig az érvényes tartományon belül maradjon!
* Oldjuk meg, hogy a tartományból kilépve $x$ koordináta esetén a szélességgel, $y$ koordináta esetén a magassággal számítunk maradékot!

**3. Konvolúciós szűrők**

* Készítsünk egy konvolúciós szűrőt, amely kiemeli az élen a képeket! (Például Sobel vagy Roberts operátorral.)
* Implementáljuk az eljárást szekvenciálisan és OpenCL segítségével!
* Végezzünk méréseket ezek összehasonlítására különböző méretű képekkel!

