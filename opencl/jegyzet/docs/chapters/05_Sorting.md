# Rendezőalgoritmusok

*Célok*

* Minimalizálni szeretnénk a számítási lépések számát.
* Maximalizálni szeretnénk a számítási egységek kihasználtságát.

*Szempontok*

* Számítási egységek száma a rendezendő elemek számához képest.
* GPU memória limit.
* Memória mozgatás költsége.

!!! task "További szempontok"

    Soroljunk fel további szempontokat!

<!--
* Memóriaelérés, összehasonlítások, cserék számítási ideje.
* Meglévő rendezettség kihasználása.
-->

*Következmények*

* Nagy adathalmazokat csak független részekre bontva tudunk rendezni.
* Ha az $n / p$ arány nagy, akkor hatékony szekvenciális algoritmusra van szükség, amely $p$ számú magon, $\left\lfloor \dfrac{n}{p} \right\rfloor$ méretű problémán fut egyidejűleg.

!!! task "Adatátviteli idők"

    Adjunk becslést az adatok mozgatásának időigényére (egy adott konfiguráción)!

## Összefésülés

Részenként rendezett sorozatok esetében szükségszerű művelet.

Kompromisszumot kell kötni!

* Összefésülés helyben. $\Rightarrow$ Lineáris beszúrási idő.
* Hatékony összefésülés. $\Rightarrow$ Lineáris méretű plusz tár.

!!! task "Legrosszabb esetek"

    Adjuk meg a legrosszabb esetét a helyben történő összefésülésnek!

## *"Oszd meg és uralkodj!"* algoritmusok

A párhuzamos architektúrából természetesen adódik, hogy a problémát kisebb, egymástól függetlenül megoldható részekre kell bontatni.

!!! task "QuickSort, MergeSort"

    Mi jelenthet problémát a Gyorsrendezés és az Összefésülő rendezés esetében?

## Páros-páratlan rendezés

A Buborékrendezés lépéseit időben egyszerre is elvégezhetjük.

* Páros indexű elemek összehasonlítása (és cseréje) a jobb oldali szomszédjukkal.
* Páratlan indexű elemekre hasonlóképpen.

Ezt a 2 lépést váltogatva minden elem a helyére kerül $n$ lépést követően.

!!! task "Implementáció"

    Adjunk egy egyszerű implementációt, majd vizsgáljuk meg a tulajdonságait!

!!! warning "Processzorszám limit"

    Hogy ha a bemenetek száma több, mint a processzorok száma, az az $n / p$ értékkel növeli a számítási időt!

## Rendezés hálózatokkal

Rögzített számú bemenet esetén a rendezési problémát hálózat formájában is reprezentálhatjuk.

* [https://en.wikipedia.org/wiki/Sorting_network](https://en.wikipedia.org/wiki/Sorting_network)

!!! task "Implementáció"

    Vizsgáljuk meg az implementációs lehetőségeket (pl.: OpenCL esetében)!

A `shuffle` és `shuffle2` függvények:

* [shuffle](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/shuffle.html)

### Batcher féle páros-páratlan összefésülő rendezés

* [https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort](https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort)

### Pairwise rendezés

* [https://en.wikipedia.org/wiki/Pairwise_sorting_network](https://en.wikipedia.org/wiki/Pairwise_sorting_network)

## Bitonikus rendezés

Tekintsünk egy $A$ valós sorozatot! Ezt monoton növekvőnek nevezzük, hogy ha bármely $i$ esetén teljesül, hogy $a_i \leq a_{i+1}$.

**Bitonikus sorozat**: A sorozat első fele monoton növekvő, a második pedig monoton csökkenő.

!!! note "Általánosítás"

    A sorozatot szintén bitonikusnak tekintjük, hogy ha az első fele csökkenő és a második fele növekvő.

**Feltételes csere**

* Legyen adott egy $n$ elemű, bitonikus sorozat.
* Az egyszerűség kedvéért feltételezzük, hogy $n$ páros.
* Hasonlítsuk össze az $a_i$ és az $a_{i + n/2}$ elemeket (ahol $i < n/2$).
* Hogy ha az $a_i \leq a_{i + n/2}$ nem teljesül, akkor cseréljük meg az elemeket.

*Eredmény*

* A kapott sorozat nem lesz bitonikus, viszont az első és második fele igen.
* A sorozat első felének minden eleme kisebb lesz, mint a második felének elemei.
* Rekurzívan végrehajtva egy rendezést kapunk.
* A feltételes cserék időben egyszerre végrehajthatók.

**Bitonikus sorozat létrehozása**

* Minden 2 elemű sorozatot bitonikus sorozat.
* Két 2 elemű sorozatból 4 elemű bitonikus sorozatot kapunk, hogy ha az 1-2, 3-4 indexeket összehasonlítjuk, és szükség esetén cseréljük.
* Két 4 elemű sorozatból 8 elemű bitonikus sorozatot úgy kaphatunk, hogy ha a 4 elemen az 1-3, 2-4 és az 1-2, 3-4 feltételes cseréket hajtjuk végre (annak függvényében, hogy növekvő vagy csökkenő részsorozatot szeretnénk kapni).

**Számítási bonyolultság**

A kapott rendezés $\mathcal{O}(\log^2(n))$ idejű.

* [https://en.wikipedia.org/wiki/Bitonic_sorter](https://en.wikipedia.org/wiki/Bitonic_sorter)

## Shell rendezés

A Shell rendezésnek többek között az alábbi előnyei vannak.

* Hatékonyan mozgatja az egymástól távol lévő elemeket.
* Tetszőleges nagy méretű bemenetek esetében működik.
* Aránylag egyszerűen implementálható.

!!! warning "Növekmények kiválasztása"

    Nem egyértelmű, hogy melyik a megfelelő sorozat a rendezéshez.

* [https://en.wikipedia.org/wiki/Shellsort](https://en.wikipedia.org/wiki/Shellsort)

## Radix (számjegyes) rendezés

A számjegyenkénti rendezéssel is független részekre bonthatjuk a problémateret.

* Feltételezzük, hogy amit rendezünk az hatékonyan számjegyekre bontható.

## Leszámláló rendezés

* [https://www.uni-miskolc.hu/~matip/parallel/pages/bsc/topic_06.html#leszamlalo-rendezes](https://www.uni-miskolc.hu/~matip/parallel/pages/bsc/topic_06.html#leszamlalo-rendezes)

!!! task "Összehasonlítás"

    Hasonlítsuk össze a rendezést a naiv, páros-páratlan szomszédos elemek cseréjére épülő algoritmussal!

## Feladatok

**1. Sorozatok generálása**

Generáljunk nagy méretű véletlenszerű elemeket tartalmazó sorozatokat.

* A *seed* érték segítségével oldjuk meg, hogy az eredmény reprodukálható legyen!

**2. Segédfüggvények**

Készítsünk segédfüggvényeket, amelyekkel ellenőrízhetők az alábbiak.

* A sorozat rendezett.
* Két sorozat azonos értékeket tartalmaz.

**3. Összefésülés helyben**

Implementáljuk az Összefésülés algoritmusát úgy, hogy a művelet helyben (plusz tár használata nélkül) kerüljön végrehajtásra!

**4. QuickSort**

Készítsünk egy OpenCL kernel függvényt, amelyik egy részintervallumon Gyorsrendezéssel rendez!

**5. Bitonic sort**

* Tekintsünk át néhány elérhető Bitonikus rendezés implementációt! (Például: [https://github.com/icaromagalhaes/opencl-bitonic-sort](https://github.com/icaromagalhaes/opencl-bitonic-sort))
* Próbáljunk meg sajátot készíteni!

**6. Shell rendezés**

Implementáljuk a Shell rendezést OpenCL segítségével!

* Vizsgáljuk meg a futási időt különböző bemenetek, bemenetméretek, problématér felbontás, növekmény sorozat esetében!

**7. Leszámláló rendezés**

Implementáljuk OpenCL segítségével a leszámláló rendezést!

* Vizsgáljuk meg annak a lehetőségét, hogy az utolsó, lineáris lépést a CPU hajtsa végre!

**8. AKS hálózatok**

Nézzünk utána az AKS hálózatoknak!
