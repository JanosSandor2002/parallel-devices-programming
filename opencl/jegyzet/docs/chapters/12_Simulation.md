# Fizikai szimulációk

## Végeselem módszerek

* *Finite Element Method*, *Finite Element Analysis*
* [https://en.wikipedia.org/wiki/Finite_element_method](https://en.wikipedia.org/wiki/Finite_element_method)

Alkalmas lehet például az alábbiak modellezésére:

* folyadékok,
* tűz, füst,
* rezgések (hang), elektromágneses hullámok (fény) terjedése,
* áramlás.

Alkalmas két- vagy többdimenziós szimuláció készítésére is.

## Részecske alapú módszerek

Egy részecskét jellemezhetnek például az alábbiak.

* Középpont (*center*): $(c_x, c_y) \in \mathbb{R}^2$
* Sugár (*radius*): $r \in \mathbb{R}$
* Sebesség (*velocity*): $(v_x, v_y) \in \mathbb{R}^2$
* Tömeg (*mass*): $m$

Egy egyszerű fizikai modellre egy részecskékre a következők hatnak:

* gravitáció,
* más részecskékkel vagy fallal való ütközésből származó erő,
* tehetetlenség.

!!! note "Erők számítása"

    Az előbbieket egymástól függetlenül számíthatjuk, majd a részecskére ható erőt ezek eredőjeként kapjuk.

Számítások optimalizálása

* A részecskékre alkalmazhatunk valamilyen térpartícionálási módszert a hatékonyabb ütközésvizsgálat érdekében.
* A részecskéket külön számítási egységekhez rendelhetjük (akár index alapján).
* Feltételezhetjük, hogy minden részecske állapotát csak egyszerre tudjuk frissíteni. (A következő iteráció számításához minden adat egyidejűleg elérhető, így a részecskék függetlenül kezelhetők.)

## Cella alapú módszerek

A szimulációban a cellák helyzetét rögzítettnek tekintjük, és a cellákhoz tartozó jellemzőket változtatjuk.

* *Cellular Automaton*
* A cellákon belül folytonosak lehetnek az értékek.
* A szabályoknak a szomszédos cellák viszonylatában elegendő működniük.
* [https://www.youtube.com/watch?v=XfhNlTt3zeQ](https://www.youtube.com/watch?v=XfhNlTt3zeQ)

!!! note "Párhuzamosítás"

    A párhuzamosításra itt is az ad lehetőséget, hogy feltételezhetjük, hogy minden cella állapotát az előző iteráció alapján egymástól függetlenül számíthatjuk.

## Háromszögekre bontás

A szimulációkban a celláknak nem szükségszerű négyzeteseknek lenniük.

* *Triangulation*
* [https://en.wikipedia.org/wiki/Surface_triangulation](https://en.wikipedia.org/wiki/Surface_triangulation)
* [https://en.wikipedia.org/wiki/Mesh_generation](https://en.wikipedia.org/wiki/Mesh_generation)

## Delaunay háromszögesítés

* Boris Delaunay, 1934.
* *Flip method*
* Bowyer-Watson algoritmus

Forrás:

* [https://en.wikipedia.org/wiki/Delaunay_triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation)
* [https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm](https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm)

## Kérdések

* Milyen előnyei és hátrányai lehetnek a folyadékok részecske és cella alapú módszerrel történő szimulációjának?
* Részecske alapú szimulációnál miből adódik a *remegés*? Hogyan lehet kiküszöbölni?
* Milyen alkalmazási területei lehetnek (a szimuláción túl) a háromszögesítésnek?
* Milyen komplexitást tudunk elérni (elvi szinten) térpartícionálási módszer használatával az ütközésvizsgálat optimalizálása érdekében?

## Feladatok

**1. Részecske alapú folyadék szimuláció**

* Definiáljuk a részecskét, mint struktúrát C-ben!
* Tegyük megjeleníthetővé a részecskerendszer állapotát!
* Implementáljuk a részecskék mozgásának a számításait!
* Végezzünk méréseket részecskeszám függvényében egy iteráció számítási idejére vonatkozóan!
* Térpartícionálási módszerrel tegyük hatékonyabbá a részecskék ütközésének számítását!
* Oldjuk meg, hogy OpenCL segítségével több részecske új pozícióját is egyidejűleg tudjuk számolni!

**2. Cella alapú folyadék szimuláció**

* Definiáljunk egy diszkrét rácsfelbontást!
* Adjunk meg szabályokat, amelyek figyelembe tudják venni, hogy egy cellában mennyi folyadék van, és hogy hat rá a gravitáció!
* Tegyük megjeleníthetővé az eredményeket!
* A cellákhoz tartozó számításokat végezzük el párhuzamosan OpenCL segítségével!

