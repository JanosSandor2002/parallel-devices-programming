# Memóriakezelés

## Adattípusok

<!-- Act. 94 -->

A C nyelv szokásos típusai, megkötésekkel és kiegészítésekkel.

### Skalár típusok

|Típus|Leírás|
|-----|------|
|`bool`|Logikai érték: `false` (0), `true` (1)|
|`char`|Előjeles 8 bites egész|
|`uchar`|Praktikusan a byte típus|
|`short`|16 bites egész|
|`int`|32 bites egész|
|`long`|64 bites egész|
|`half`|16 bites lebegőpontos érték, IEEE-754-2008|
|`float`|32 bites lebegőpontos érték, IEEE-754|
|`intptr_t`|Mutatóvá konvertálható előjeles egész|
|`uintptr_t`|Mutatóvá konvertálható előjel nélküli egész|
|`size_t`|A `sizeof` operátor által visszaadott típus|
|`void`|Hiányzó, tetszőleges típus|

* Szigorúbb/szabványosított a bitek száma.
* Az egészekből van `unsigned` típus, ami írható `u` prefixszel.

!!! warning "Byte sorrend"

    Az OpenCL szabvány sajnos nem tudta rögzíteni a byte sorrendet a különféle eszközök miatt.

### Dupla pontosság

A `double` típus általánosan nem tekinthető támogatottnak. Bővítmény formájában elérhető:

```c
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
```

!!! note "Bővítmények bekapcsolása"

    Az összes bővítményt egyidejűleg az `all` néven aktiválhatjuk.

A kernel kódjában a következő formában ellenőrízhetjük, hogy elérhető-e a `double` típus.
```c
#ifdef FP_64
double x = 0.0;
#else
float x = 0.0;
#endif
```

Kerekítési módok:

* a legközelebbi értékhez,
* mínusz végtelenhez,
* plusz végtelenhez,
* 0-hoz (*truncation*).

### Fél pontosság

* A `half` a `float` és a `double` típusoknál újabb típus.
* Az Nvidia vezette be a Cg nyelv részeként.
* Az OpenGL és a Direct3D is átvette.
* A `cl_khr_fp16` bővítmény jelzi, hogy elérhető-e.

Részei:

* 1 bit: előjel bit
* 5 bit: exponens (15-tel eltolva)
* 10 bit: törtrész

### Vektor típusok

A következő típusok vektorként használhatók, hogy ha kiegészítjük a típus nevét a 2, 3, 4, 8, 16 számértékek valamelyikével.
```c
char short int long uchar ushort uint ulong float
```

### Inicializálás

Hasonlóképpen inicializáható, mint a tömbök, csak kerek zárójellel:

```c
float4 coord = (float4)(1.0, 2.0, 3.0, 1.0);
```

Vektorok segítségével is inicializálható.

```c
float2 a = (float2)(1.0, 2.0);
float2 b = (float2)(3.0, 1.0);
float4 coord = (float4)(a, b);
```

!!! note ""

    Skalárokkal együtt vegyesen is használhatók.

### Komponensek elérése

Indexelés `.sN` formában (ahol `N` egy hexadecimális számjegy)

```c
coord.s2
```

Név szerinti elérés

```c
coord.x, coord.y, coord.z, coord.w
```

Részek szerinti elérés

```c
coord.hi, coord.lo, coord.even, coord.odd
```

Több komponens egyidejű hivatkozása

```c
coord.yw
coord.s03
```

Az így kijelölt tartományoknak az inicializáláshoz hasonlóan adható meg egyidejűleg több érték is.

### Preferált vektor méretek

A *device*-nak típusonként (a hardveres kivitelezésből adódóan) van preferált mérete a vektorok elemszámára vonatkozóan.

```c
cl_uint preferred_width;
clGetDeviceInfo(
    device,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
    sizeof(preferred_width),
    &preferred_width,
    NULL
);
```

* A preferált mérethez igazodás előnyös számítási teljesítmény szempontjából.
* A *host* kódban, fordításnál paraméterként átadhatjuk a preferált méretet, például a `-DPREF_VECT_SIZE_16` esetén:

```c
#if PREF_VECT_SIZE_16
// ...
#endif
```

* Külön kerneleket is létrehozhatunk *device*-nak megfelelően (amelynek a nevében jelezhetjük a preferált méretet).

## OpenCL C fordító

<!-- Act. 55 -->

```c
clBuildProgram(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const char *options,
    void (CL_CALLBACK *pfn_notify)( cl_program program, void *user_data),
    void *user_data)
)
```

* [clBuildProgram](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clBuildProgram.html)

!!! note "Példa"

    Nézzük meg az `examples/02_compilation` példát!

!!! task "Típusok"

    Próbáljuk ki, hogy valóban működnek az előzőekben bemutatott típusok!

### Fordító opciók

|Opció|Leírás|
|-----|------|
|`-cl-std=VERSION`|OpenCL verzió beállítása|
|`-D NAME`|Makró beállítás 1 értékre|
|`-D NAME=VALUE`|Makró beállítás `VALUE` értékre|
|`-Werror`|Minden figyelmeztetést hibának tekint|

!!! task "Makró átadása"

    Nézzük meg, hogy valóban át tudunk adni értéket fordító opción keresztül!

### Program információk

```c
cl_int clGetProgramInfo(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret
)
```

* [clGetProgramInfo](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetProgramInfo.html)

!!! task "Bináris mérete"

    Kérdezzük le az elkészített binárisok számát és méretüket!

### Fordító kimenete

```c
cl_int clGetProgramBuildInfo(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret
)
```

* [clGetProgramBuildInfo](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetProgramBuildInfo.html)

!!! task "Hibák kiíratása"

    Hibás fordítás esetén írassuk ki, hogy mi a probléma!

## Hibakeresés

<!-- Het. 269 -->

A kernel kódján belül használhatjuk a `printf`-et.

* A `clFinish` meghívás hatására kerül kiíratásra a kernelekhez tartozó kimenet. (Ez magától is meghívódik.)
* A kiírások sorrendje egyáltalán nem garantált.

!!! task "My ID"

    Írassuk ki a kerneleken belül a saját azonosítóikat!

## Profilozás

<!-- Het. 252 -->

A profilozás események segítségével valósítható meg.

Lépései:

* Be kell állítani a `clCreateCommandQueue` parancsban a `CL_QUEUE_PROFILING_ENABLE` property-t.
* Meg kell adni egy `cl_event` eseményt a profilozandó művelethez.
* Le kell kérdezni a profilozás eredményét a `clGetEventProfilingInfo` függvénnyel.

```c
cl_int clGetEventProfilingInfo(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret
)
```

* [clGetEventProfilingInfo](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetEventProfilingInfo.html)

Beállítható `cl_profiling_info` paraméterek:

|Név|Leírás|
|---|------|
|`CL_PROFILING_COMMAND_QUEUED`|A parancs bekerült a sorba|
|`CL_PROFILING_COMMAND_SUBMIT`|A parancs kiküldésre került a *device*-ra|
|`CL_PROFILING_COMMAND_START`|A parancs végrehajtása elkezdődött|
|`CL_PROFILING_COMMAND_END`|A parancs végrehajtása befejeződött|

A lekérdezett paraméter értéke `cl_ulong` típusként, nanoszekundumban van megadva.

!!! warning "Blokkolás"

    A `clFinish` használatára szükség lehet, hogy a profilozandó parancs le tudjon futni a lekérdezés előtt!

!!! note "Példa"

    Nézzük meg az `examples/03_profiling` példát!

## Buffer létrehozása

<!-- Act. 68 -->

```c
cl_mem clCreateBuffer(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void *host_ptr,
    cl_int *errcode_ret
)
```

* [clCreateBuffer](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateBuffer.html)

A `cl_mem_flags` lehetséges értékei:

|Érték|Leírás|
|-----|------|
|`CL_MEM_READ_WRITE`|Olvasható és írható|
|`CL_MEM_WRITE_ONLY`|Csak írható|
|`CL_MEM_READ_ONLY`|Csak olvasható|
|`CL_MEM_USE_HOST_PTR`|A *device* a *host* memóriáját használja|
|`CL_MEM_COPY_HOST_PTR`|A *device* másolatot készít a *host* memóriájáról|
|`CL_MEM_ALLOC_HOST_PTR`|Egy nem kilapozható memóriát allokál|

!!! note "Memória flag-ek"

    A bufferhez megadott flag-ek (`cl_mem_flags`) a device-ra vonatkoznak.
    A helytelen hozzáférés nem definiált működést eredményez.

* A `CL_MEM_READ_WRITE` az alapértelmezett érték.
* A `_HOST_PTR` végződésűek az allokáció módját írják le.
* A *flag*-eket kombinációban lehet használni, például: `CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR`.
* A másolás számítási időt, plusz memória allokálást, de gyorsabb elérést eredményez.
* A `CL_MEM_ALLOC_HOST_PTR` a `CL_MEM_COPY_HOST_PTR`-el együtt használható.
* A csak írható memória esetében nem kell beállítani a `host_ptr` értéket (lehet `NULL`).

### Részbufferek létrehozása

A bufferek bizonyos részeit más kernelek számára is elérhetővé tudjuk tenni.

```c
cl_mem clCreateSubBuffer(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void *buffer_create_info,
    cl_int *errcode_ret
)
```

* [clCreateSubBuffer](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateSubBuffer.html)

Paraméterei:

* A `flags` megadása megegyezik a buffer létrehozásánál használttal.
* A `buffer_create_type` alapvetően `CL_BUFFER_CREATE_TYPE_REGION`.
* A `buffer_create_info` egy `cl_buffer_region` struktúrára mutat, amely gyakorlatilag egy *offset* és *méret* párt ír le.

Például egy `int` típusú elemeket tartalmazó bufferből a `[400, 499]` indextartomány kijelölése így adható meg:

```c
cl_buffer_region region;
region.origin = 300 * sizeof(int);
region.size = 100 * sizeof(int);
```

## Buffer adatok lekérdezése

<!-- Act. 75 -->

A bufferek adatait a `clGetMemObjectInfo` függvénnyel kérdezhetjük le.

```c
cl_int clGetMemObjectInfo(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret
)
```

* [clGetMemObjectInfo](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetMemObjectInfo.html)

A `param_name` néhány lehetséges értéke:

|Név|Leírás|
|---|------|
|`CL_MEM_FLAGS`|A memória objektumhoz tartozó *flag*-eket adja vissza|
|`CL_MEM_HOST_PTR`|A *host* memóriaterület címét adja vissza (`void*` típussal)|
|`CL_MEM_SIZE`|A memóriaterület méretét adja vissza (`size_t` típussal)|

!!! note "Példa"

    Nézzük meg az `examples/04_buffers` példát!

## Command Queue parancsok

<!-- Act. 77 -->

A *host* és a *device* között az adatokat a `clEnqueueReadBuffer` és `clEnqueueWriteBuffer` parancsokkal lehet mozgatni:

* [clEnqueueReadBuffer](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueReadBuffer.html)
* [clEnqueueWriteBuffer](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueWriteBuffer.html)

A létrehozott bufferek között is lehet másolási műveletet végrehajtani a `clEnqueueCopyBuffer` paranccsal:

* [clEnqueueCopyBuffer](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueCopyBuffer.html)

## Feladatok

**1. Hibakódok**

* Keressük meg, hogy mely fájlban kerültek definiálásra az OpenCL hibakódok!
* Definiáljunk egy függvényt, amelyik visszaadja a hibakód nevét!

**2. Hibakezelés**

Készítsünk programokat például a következő hibalehetőségekre:

* túl nagy méretű kernel program,
* 0-val osztás,
* OpenCL C nyelvi limitációk.

Vizsgáljuk meg a kimeneteket!

* Próbáljuk meg kiszervezni a kernel fordításához, fordító kimenetének eléréséhez szükséges programrészeket egy függvénybe!

**3. Profilozás**

Készítsünk programot az egyes programrészek futási idejének méréséhez!

* Válasszunk megfelelő mértékegységet!
* Jelenítsük meg grafikonon, hogy a program mennyi időt tölt az egyes műveletekkel a `01_hello` példa esetében!

**4. Mátrix műveletek**

Implementáljuk a következő mátrix műveleteket OpenCL segítségével!

* Transzponálás
* Szorzás
* Oszlopösszeg számítás
* Sorösszeg számítás

(*A mátrix megadásához használjunk sor- vagy oszlopfolytonos tárolási módot.*)

**5. Profilozás, optimalizálás**

Az eddigi számítási feladatok esetében végezzünk méréseket és optimalizáljuk a programot!

* Próbáljunk meg lokális és privát memória használatával javítani a számítási teljesítményt!
* Optimalizáljuk a munkacsoportok és munkaelemek számát!
* Mérjük le az adatok mozgatásához szükséges időt!
* Készítsünk méréseket a komplexitás becsléséhez!
* Ábrázoljuk, összegezzük a kapott eredményeket!
