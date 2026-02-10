# Szinkronizálás

Szinkronizálásra OpenCL esetében több helyen is szükség lehet.

* Host és device kód között. &rarr; Eseményekkel oldjuk meg.
* Command Queue-k között. &rarr; Csak említjük.
* Work item-ek között. &rarr; Barrier-ekkel, és atomi műveletekkel oldjuk meg.

## Események kezelése

<!-- Act. 163 -->

Az eseményeknek 3 fő felhasználási területe van:

* Host oldal értesítése egy eseményről,
* parancsok szinkronizálása,
* profilozás.

!!! warning "Host értesítése"

    Egy parancs kiadását (sorba való beszúrását) követően a host nem tudja már befolyásolni annak a végrehajtását.

* Az események adatait `cl_event` típusú struktúrák tartalmazzák.
* A parancsok kiadásakor a társított esemény címe utolsó paraméterként szerepel.

A callback függvény szignatúrája:
```c
void CL_CALLBACK op_complete(cl_event e, cl_int status, void* data);
```

Callback beállítása:
```c
cl_int clSetEventCallback(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK *pfn_event_notify) (
        cl_event event,
        cl_int event_command_exec_status,
        void *user_data
    ),
    void *user_data
)
```

!!! note "Példa"

    Nézzük meg az `examples/05_events` példát!

Amennyiben egyidejűleg több *device* lenne, és emiatt több *command queue*-t is használnánk, úgy a szinkronizációhoz az eseményekből várólistákat (*waiting lists*) hozhatnánk létre.

## Work-item szinkronizáció

* Work item-ek szinkronizációjára akkor lehet szükség, hogy ha azonos adatterületen dolgoznak.
* Work item-eket csak work group-on belül lehet szinkronizálni.

### Barrier-ek

Barrier segítségével bevárhatjuk, hogy az összes többi work item is befejezze a végrehajtást az adott pontig.

!!! info "`barrier`, `work_group_barrier`"

    A `barrier` függvény az új OpenCL változatokban `work_group_barrier` néven érhető el.

```c
void work_group_barrier(
    cl_mem_fence_flags flags
)
```

A `cl_mem_fence_flags` lehetséges értékei:

* `CLK_LOCAL_MEM_FENCE`: alapértelmezés szerint a work group scope-ját veszi figyelembe,
* `CLK_GLOBAL_MEM_FENCE`: csak a `scope` paraméter megadásával használható.

```c
first_part();
work_group_barrier(CLK_LOCAL_MEM_FENCE);
second_part();
```

* [work_group_barrier](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/work_group_barrier.html)

### Atomi műveletek

<!-- Act. 184 -->

!!! task "*Lost Update* probléma"

    Tegyük fel, hogy az `x += 3` műveletet szeretnénk végrehajtani 2 work item-mel párhuzamosan.
    Vizsgáljuk meg a felmerülő problémát!

Az OpenCL-ben elérhető atomi műveletek:

* 1 változós: `atomic_inc`, `atomic_dec`
* 2 változós: `atomic_add`, `atomic_sub`, `atomic_max`, `atomic_min`, `atomic_xchg`
* 3 változós: `atomic_cmpxchg`

Mindegyik függvény mutatót vár első paraméterként. Az eredmény abba kerül.

Az *"Összehasonlítás és csere"* művelet
```c
int atomic_cmpxchg(int* p, int cmp, int value);
```

A `p` által hivatkozott értéket kicseréli `value`-ra, hogy ha `*p == cmp`. Röviden:
```c
*p = ((*p == cmp) ? value : *p)
```

Mutex megvalósítása
```c
#define LOCK(m) atom_cmpxchg(m, 0, 1)
#define UNLOCK(m) atom_xchg(m, 0)
```

Használata
```c
__kernel void mutexed(__global int* mutex, __global int* sum)
{
    while (LOCK(mutex));
    *sum += 1;
    UNLOCK(mutex);
}
```

!!! warning "Workgroup-ok közötti szinkronizáció"

    Globális változót lehetne használni mutex-nek, spinlock-os szinkronizációhoz.
    Az egyidejűleg futtatható work item-ek száma viszont limitált, ezért live lock alakulhat ki!

## Termelő-fogyasztó probléma

Egyszerűbb modellben a CPU termeli a feladatokat, a *work item*-ek pedig megoldják azokat.

A probléma megoldható csővezetékek (*pipes*) használatával.

* Egy speciális memória objektum. (Hasonló, mint a buffer.)
* FIFO adatszerkezet.
* Önmagában kezeli a zárolást. (Saját magunknak körülményes lenne implementálni.)
* Kontextushoz kötött.

```c
cl_mem clCreatePipe(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties * properties,
    cl_int *errcode_ret
)
```

* [clCreatePipe](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreatePipe.html)

## Adatfolyam feldolgozás

* *Multiple Instruction Multiple Data*
* A nagy adathalmazt részekre bontva, azokat adatfolyamnak tekintve oldjuk meg.

!!! warning "Adatbetöltés"

    Az adatfolyam feldolgozásnál feltételezhető, hogy a teljes adathalmaz nem fog egyszerre elférni a memóriában.

## Cella alapú számítások

A problémák egy részét egymással összekapcsolt, kommunikációra képes számítási egységek rendszerével is megoldhatjuk.

A kernelek kódja szekvenciális, így gyakorlatilag a CSP modellel is dolgozhatunk:

* [https://en.wikipedia.org/wiki/Communicating_sequential_processes](https://en.wikipedia.org/wiki/Communicating_sequential_processes)

## Feladatok

**1. Események**

* Adjunk hozzá eseményt a kernel végrehajtásának ellenörzéséhez is!
* Készítsünk a buffer kiolvasásához egy olyan *callback* függvényt, amelyik a bufferből visszaolvasott értékekkel hívódik meg!

**2. Gyakoriságok számítása**

* Generáljunk egy tömböt, amely [0, 100] intervallumon tartalmaz egész értékeket!
* OpenCL segítségével határozzuk meg az elemek gyakoriságát!

**3. Szórás számítása**

* Hozzunk létre véletlenszerű értékekkel egy tömböt!
* A tömb mérete legyen paraméter!
* Számítsuk ki a tömbben lévő értékek szórását OpenCL segítségével!

**4. Adatfájl feldolgozása**

* Készítsünk elő egy néhány száz megabyte-os adatfájlt!
* OpenCL segítségével számítsuk ki, hogy mennyi 0-ás byte van benne!
* Mérjük le az adatbetöltési és feldolgozási időket!
* Optimalizáljuk a programot a hatékonyabb végrehajtás érdekében!
* Ábrázoljuk a kapott eredményeket!
