# Összekapcsolás az OpenGL-el

## Az OpenGL és OpenCL viszonya

Az OpenGL 2-es verziójában megjelent a *shader* (*árnyaló*) programozás. Ezek hasonlóak az OpenCL kernelekhez. Az OpenCL előnyei:

* Több függvényt támogatnak. (Általánosabb az API.)
* A kernelek hozzáférnek a privát és a lokális memóriához.
* A *work-item*-ek szinkronizált módon adatokat oszthatnak meg egymás között.

Az OpenGL és az OpenCL más célból készült. Más eszközöket és terminológiát használ.

|OpenGL|OpenCL|
|------|------|
|Vertex Buffer Object|Buffer Object|
|Texture Object|Image Object|

Egy jelentős előny, hogy az adatokat *device* oldalon egyaránt el lehet érni.

!!! warning "Minimális API verzió"

    Az OpenCL-el való kapcsolódáshoz legalább OpenGL 3.3-as verzióra van szükség.

Forrás:

* [https://www.khronos.org/opengl/wiki/History_of_OpenGL](https://www.khronos.org/opengl/wiki/History_of_OpenGL)
* [https://registry.khronos.org/OpenGL-Refpages/gl4/](https://registry.khronos.org/OpenGL-Refpages/gl4/)

### A kapcsolat felépítése

Az OpenCL kontextus létrehozásakor be kell hivatkozni OpenGL kontextust.

!!! warning "Hivatkozási mód"

    Az OpenGL kontextus hivatkozása OS függő.

Windows alatt (`windows.h` fejlécből):

* `wglGetCurrentContext` $\rightarrow$ `HGLRC` *OpenGL rendering context* $\rightarrow$ `GL_CL_CONTEXT_KHR` property
* `wglGetCurrentDC` $\rightarrow$ `HDC` *Device Context* $\rightarrow$ `CL_WGL_HDC_KHR`

```c
cl_context_properties properties[] = {
    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
    0
};
context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
```

### Adatok megosztása

Ahhoz, hogy egy OpenGL buffert el lehessen érni OpenCL-ből, azt a `clCreateFromGLBuffer` függvénnyel kell OpenCL-hez kapcsolni.

* [clCreateFromGLBuffer](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateFromGLBuffer.html)

*Image* objektum hasonlóképpen érhető el OpenGL textúrából a `clCreateFromGLTexture2D` függvénnyel.

### Az alkalmazás lépéseinek sorrendje

Egy egyszerű alkalmazás például az alábbi lépéseket hajthatja végre:

* OpenGL inicializálása,
* OpenCL inicializálása,
* hozzáférés az OpenGL adatszerkezetekhez OpenCL-ből,
* OpenCL kernelek futtatása,
* grafika megjelenítése.

A kontextus kezelésénél figyelni kell arra, hogy a `glFinish` és a `clFinish` függvények is meghívásra kerüljenek.

## Sugárkövetés

* A képpontok színét egymástól függetlenül ki lehet számítani.
* Mivel jellemzően elég sok képpont van, ezért a művelet számításigényes.

### Inverz fényútkövetés

A sugarakat a kamerából indítjuk, és azt vizsgáljuk, hogy milyen módon tudnak eljutni a fényforrásig.

Egy sugarat a $\textbf{p}(t) = \textbf{o} + t \cdot \textbf{d}$ formában írhatunk fel, ahol

* $\textbf{p}(t)$: a sugár egy pontja a $t$ paraméterértéknél,
* $t \in \mathbb{R}, t > 0$: a futó paraméterünk,
* $\textbf{o}$: a fénysugár kiinduló pontja (*origin*),
* $\textbf{d}$: a fénysugár iránya (*direction*).

### Gömbök metszése

A gömb egyenlete: $x^2 + y^2 + z^2 = R^2$, ahol $R$ a gömb sugara.

Implicit alak a $\textbf{p}(t)$ paraméterrel: $(\textbf{p}(t))^2 - R^2 = 0$.

!!! task "Négyzetes alak"

    Helyettesítsünk vissza, majd rendezzük át az egyenletet négyzetes alakra!

!!! task "Megoldás"

    Oldjuk meg az egyenletet $t_{1,2}$-re!

!!! task "Megoldások száma"

    Vizsgáljuk meg, hogy mennyi megoldása lehet az egyenletnek, és az milyen esetet takar!

A legkisebb nemnegatív $t$ értéket kell választanunk!

### Normál vektor számítása

Tegyük fel, hogy ismerjük a $\textbf{p}$ pontot, ahol a sugár (először) metszi a gömböt.

A normálvektort a következő összefüggéssel kapjuk: $\textbf{n} = \dfrac{\textbf{p} - \textbf{o}}{|\textbf{p} - \textbf{o}|}$.

!!! note "Normalizálás"

    A normálvektornak nem lenne szükségszerű egységnyi hosszúságúnak lennie, de a számításainkat egyszerűsíti ez a feltételezés.

### Visszaverődés számítása

Az egyszerűség kedvéért tekintsük az ideális fényvisszaverődés esetét. Ekkor

$$
\textbf{r} = \textbf{d} - (2 \cdot \textbf{d} \cdot \textbf{n}) \cdot \textbf{n},
$$

ahol $\textbf{r}$ a visszaverődő fénysugár irányának a vektora.

!!! warning "Eltolás"

    A gyakorlatban egy $\varepsilon$ eltolást szoktak alkalmazni a sugárra, hogy ne *ragadjon* be a kiinduló pontba a visszaverődés során.

Forrás:

* [https://raytracing.github.io/books/RayTracingInOneWeekend.html](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

### Kép mentése fájlba

* *Portable Pixel Map*, *Portable PixMap*
* [https://en.wikipedia.org/wiki/Netpbm](https://en.wikipedia.org/wiki/Netpbm)
* [https://netpbm.sourceforge.net/doc/ppm.html](https://netpbm.sourceforge.net/doc/ppm.html)

## Kérdések

* Milyen előnyei és hátrányai vannak az OpenCL nyelvnek sugárkövető írásakor a C-hez képest?

## Feladatok

**1. OpenGL-hez kötődő függvények**

* Gyűjtsük ki az OpenCL API-ból azokat a függvényeket, amelyek OpenGL-hez kötődnek!
* Hasonlítsuk össze az OpenCL és a GLSL API-ját!

**2. Sugárkövetés**

* Implementáljuk a sugárkövető algoritmus szekvenciális változatát!
* Készítsük el az OpenCL implementációt, amely kernelenként számolja ki a képponthoz tartozó színeket!
* Oldjuk meg a kamera mozgatását, és mentsük ki ennek segítségével képek sorozatát!
* Mozgassuk a gömböket!
* Implementáljuk a síkokkal való metszést!
* Rakjunk sakktábla textúrát a síkokra és a gömbökre!
* Valósítsunk meg nem ideális fényvisszaverődést!
* Egy pixelhez több fénysugarat indítva oldjuk meg az élsimítást!
* A fénysugarak hosszának követésével valósítsunk meg köd jellegű hatást!

