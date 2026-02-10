# Gépi tanulás

* A statisztikai módszerek közé tartozik.
* A tanítás paraméterhangolást jelent.
* A modell paraméterterében keressük azt a pontot, ahol a hibafüggvény minimális.

Párhuzamosításra több ok miatt is szükség lehet.

* Nagy mennyiségű mintánk van, amire hangolni kell.
* Nagy az állapottér, amit be szeretnénk járni. (Például magas dimenziószám és/vagy közel folytonos tartományok.)
* A kiértékelés/tanítás fázisai számításigényesek.
* A hangolás jellemzően sztochasztikus módszerekkel történik, amelynél a pontosság számítási idő ellenében javítható.
* Meta/hyper paramétereink vannak, amire plusz hangolást szeretnénk végezni.

Forrás:

* [https://en.wikipedia.org/wiki/Machine_learning](https://en.wikipedia.org/wiki/Machine_learning)
* [https://en.wikipedia.org/wiki/Timeline_of_machine_learning](https://en.wikipedia.org/wiki/Timeline_of_machine_learning)

## Típusai

### Felügyelt tanulás

* A tanításhoz minden bemeneti minta esetében megadjuk, hogy milyen kimenetre számítunk.
* Az *elvárt* és a *kapott* értékek alapján hangoljuk a modellt.

Jellemző problémák

* Osztályozás
* Regresszió
* Előrejelzés

### Megerősítéses tanulás

* Adott egy (virtuális) környezet (*environment*) amit az ágens (*agent*) maga próbál bejárni.
* Az ágens akciókat tud végrehajtani.
* A környezetből érkező megerősítések alapján dönti el, hogy a megelőző műveletsor hogyan értékelendő, hogyan használandó a hangoláshoz.

Jellemző problémák

* Vezérlés (irányítás)

### Felügyelet nélküli tanulás

* Megadunk bizonyos elvárásokat az eredményre vonatkozóan.

Jellemző problémának

* Klaszterezés
* Dimenzió redukció

## Mesterséges Neurális Hálózatok

* ANN: *Artificial Neural Network*
* MLP: *Multi-Layer Perceptron*
* Aktivációs függvény
* Hibafüggvény (*Loss Function*, *Error Function*)

!!! task "A számítási modell"

    Vizsgáljuk meg, hogy hogyan működik pontosan az ANN, azon belül is az előrecsatolt eset!

### Back Propagation

* A kimenet irányából terjesztjük vissza a hibát.
* Rengeteg alternatíva van arra, hogy a súlyok megváltoztatásával az adott bemenetre az elvárt kimenetet kapjuk.
* Megvizsgáljuk, hogy a súlyok változtatása hogyan járul hozzá a hibához. $\Rightarrow$ Gradienst számolunk.
* Meghatározzuk, hogy mennyit lenne érdemes módosítani a súlyokon az adott bemenetre adott eredmény javítása érdekében. ($\Delta$ mátrix, $\delta^{(l)}_{i, j}$.)
* Egy $\alpha$ tanítási tényezővel (*learning rate*) súlyozva módosítjuk a háló súlyait.

Számítási példa:

* [https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

!!! task "Párhuzamosítás"

    Milyen lehetőségek adódnak a párhuzamosításra?

## Kérdések

* Milyen viszonyban áll egymással a gépi tanulás és a mesterséges intelligencia témaköre?
* Miért használnak a tanító adathalmaz mellett validációs adathalmazt is?
* Miért tekinthető a gépi tanulás optimalizálási problémának?
* Az ANN esetében milyen aktivációs függvényeket szoktak használni?
* Mire szolgál az ANN esetében a *bias*?

## Feladatok

**1. Neurális hálózat párhuzamosított tanítása**

* Implementáljunk egy előrecsatolt neurális hálózatot!
* Gyűjtsünk/állítsunk össze tanító mintákat!
* Adjunk meg egy hibafüggvényt (például a MSE-t)!
* A háló tanításához implementáljunk egy egyszerű, sztochasztikus gradiens alapú módszert!
* Implementáljuk a *Back Propagation* algoritmust!
* OpenCL segítségével futtassuk a betanítást külön számítási egységeken!
* Mérjük le a futási időket, és határozzuk meg a gyorsítás mértékét!

