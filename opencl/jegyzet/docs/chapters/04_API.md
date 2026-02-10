# OpenCL API-k áttekintése

## A teljes OpenCL C API

* [https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html)

## C++

* [https://github.com/KhronosGroup/OpenCL-CLHPP](https://github.com/KhronosGroup/OpenCL-CLHPP)

* A `cl` névtér használata.
* `cl::string` és `cl::vector` típusok.
* Hibakezelés kivétel objektumokkal.
* Objektumhoz kötött funkciók (metódusok formájában).

!!! note "`cl` és `std` tárolók"

    [https://stackoverflow.com/questions/13787012/why-should-i-use-clvector](https://stackoverflow.com/questions/13787012/why-should-i-use-clvector)

A C elemek C++-os megfelelői

|C|C++|
|-|---|
|`cl_platform`|`cl::Platform`|
|`cl_device`|`cl::Device`|
|`cl_context`|`cl::Context`|
|`cl_program`|`cl::Program`|
|`cl_kernel`|`cl::Kernel`|

Platformok lekérdezése

```c
std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms);
```

Eszközök lekérdezése

```c
std::vector<cl::Device> devices;
platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
```

Kontextus létrehozása

```c
cl::Context context(devices);
std::vector<cl::Device> contextDevices = context.getInfo<CL_CONTEXT_DEVICES>();
```

Eszközadatok lekérdezése

```c
for (const cl::Device& device : contextDevices) {
    std::string name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Device name: " << name << std::endl;
}
```

Hibakezelés

```c
try {
    // ...
} catch (cl::Error e) {
    std::cout << e.what() << "(code: " << e.err() << ")" << std::endl;
}
```

!!! task "C++ példák"

    Tekintsük át a [https://github.com/KhronosGroup/OpenCL-CLHPP/tree/main/examples](https://github.com/KhronosGroup/OpenCL-CLHPP/tree/main/examples) oldalon lévő példákat!

!!! note "C++ AMP"

    *Accelerated Massive Parallelism*: [C++ AMP Overview](https://learn.microsoft.com/en-us/cpp/parallel/amp/cpp-amp-overview?view=msvc-170)

## Java

Több alternatív megközelítéssel is találkozni Java-ban.

### Aparapi

* [https://aparapi.com/](https://aparapi.com/)
* Az AMD adta ki 2010-ben.
* Csak AMD/ATI videókártyákkal működik.
* A kernelek definiálását magasabb szinten teszi lehetővé.

!!! task "Kernel készítése Java-ban"

    Nézzük meg, hogy hogyan tudunk kernelt definiálni és futtatni: [Getting started](https://aparapi.com/introduction/getting-started.html)!

### JavaCL

* `.jar` fájlok formájában érhető el: [https://code.google.com/archive/p/javacl/downloads](https://code.google.com/archive/p/javacl/downloads)

A C elemek JavaCL-es megfelelői

|C|JavaCL|
|-|---|
|`cl_platform`|`CLPlatform`|
|`cl_device`|`CLDevice`|
|`cl_context`|`CLContext`|
|`cl_program`|`CLProgram`|
|`cl_kernel`|`CLKernel`|

Platformok lekérdezése

```java
CLPlatform[] platforms = JavaCL.listPlatforms();
```

Eszközök lekérdezése

```java
CLDevice[] devices = platforms[0].listAllDevices(true);
```

Kontextus létrehozása

```java
CLContext context = JavaCL.createBestContext();
```

*Command Queue* létrehozása

```java
CLQueue queue = context.createDefaultQueue((CLDevice.QueueProperties[])null);
```

Program létrehozása

```java
CLProgram program = context.createProgram("... Kernel source");
```

!!! task "JavaCL demo-k"

    Tekintsük át a demo-kat: [JavaCL demo-k](https://github.com/nativelibs4java/JavaCL/tree/master/Demos/src/main/java/com/nativelibs4java/opencl/demos)!

## C\#

* A Java-hoz hasonlóan nem túl elterjedt a használata OpenCL esetében.
* [https://www.nuget.org/packages/OpenCL.Net](https://www.nuget.org/packages/OpenCL.Net)

!!! task "Vektorok összeadása"

    Tekintsük át a vektorok összeadásához készített programot: [VectorAdd.cs](https://github.com/tunnelvisionlabs/NOpenCL/blob/master/NOpenCL.Test/NVidia/VectorAdd.cs)

## Python

* A PyOpenCL tekinthető a hivatalos változatnak: [https://pypi.org/project/pyopencl/](https://pypi.org/project/pyopencl/)

A C elemek PyOpenCL-es megfelelői

|C|PyOpenCL|
|-|---|
|`cl_platform`|`Platform`|
|`cl_device`|`Device`|
|`cl_context`|`Context`|
|`cl_program`|`Program`|
|`cl_kernel`|`Kernel`|

Platformok lekérdezése

```python
import pyopencl as cl

for platform in cl.get_platforms():
    print(platform.get_info(cl.platform_info.NAME))
```

Eszközök lekérdezése

```python
devices = platform.get_devices()
```

Kontextus létrehozása

```python
context = cl.Context(devices)
```

*Command Queue* létrehozása

```python
queue = cl.CommandQueue(
    context,
    devices[0],
    cl.command_queue_properties.PROFILING_ENABLE)
```

!!! task "PyOpenCL dokumentáció"

    Tekintsük át a PyOpenCL dokumentációját: [https://documen.tician.de/pyopencl/](https://documen.tician.de/pyopencl/)!

## Feladatok

**1. OpenCL verziók**

Vizsgáljuk meg, hogy milyen változásokon ment keresztül az OpenCL az egyes verziókban!

**2. További nyelvek**

Vizsgáljunk meg további programozási nyelveket (pl.: Haskell), amelyekkel szintén lehet OpenCL-t használni.

**3. Saját API tervezés**

A C (host API) alapján próbáljon meg egy magas szintű nyelven példákat adni, hogy hogyan lenne egyszerűbb az OpenCL használata.
