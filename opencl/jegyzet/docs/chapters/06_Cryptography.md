# Kriptográfiai algoritmusok

* *Encryption-Decryption*
* *Cipher*/*Cypher*, *Ciphertext*
* DES: *Data Encryption Standard*, AES: *Advanced Encryption Standard*
* [https://en.wikipedia.org/wiki/Cryptography](https://en.wikipedia.org/wiki/Cryptography)

## Hash számítás

!!! task "Áttekintés"

    * Mi jellemzi a hash függvényeket?
    * Hol, és mire használják őket?

<!--
Jellemzői
* Gyorsan számolható.
* Nagyon kicsi az ütközés valószínűsége.
* Kis változás a bemenetben (pl.: 1 bit) nagy változást okoz a hash kódban.
* Nem invertálható (mivel kisebb elemszámú halmazra képez).
* Hash code esetében olyan halmazra képez a függvény, amelynek az elemei azonos méretűek.

Alkalmazások
* Jelszavak tárolása
* Fájl egyezőség vizsgálat (pl.: letöltésnél, fájlrendszer esetében)
* Módosítás ellenörzése üzenetek esetében.
* Hitelesítés, time stamping
* Tartalom szerinti címzés (pl.: git)
* HashTable, Map adatszerkezet
* Blokklánc korábbi elemeinek a hivatkozásához

https://en.wikipedia.org/wiki/Cryptographic_hash_function
-->

### MD5 Hash

* *Message Digest*
* RFC 1321, 1992.
* 512 bit $\rightarrow$ 128 bit

Algoritmus

* [https://en.wikipedia.org/wiki/MD5](https://en.wikipedia.org/wiki/MD5)
* [https://www.simplilearn.com/tutorials/cyber-security-tutorial/md5-algorithm](https://www.simplilearn.com/tutorials/cyber-security-tutorial/md5-algorithm)

!!! warning "Biztonság"

    Az MD5 hash manapság már nem tekinthető biztonságos hash-nek! (Integritás ellenörzésre továbbra is kiválóan alkalmas.)

### Biztonságos hash, SHA, SHA-2

* SHA: *Secure Hash Algorithm*
* Többségében az NSA (*National Security Agency*) fejlesztette ki őket.
* SHA-2: Az SHA-256 és az SHA-512 a leginkább elterjedt.

!!! task "Paraméterek"

    Hasonlítsuk össze az SHA-256 és az SHA-512 fő paramétereit!

Alkalmazási területek

* SSL (*Secure Socket Layer*)
* TLS (*Transport Layer Security*)
* PGP (*Pretty Good Privacy*)

Paraméterek

* [https://en.wikipedia.org/wiki/Secure_Hash_Algorithms](https://en.wikipedia.org/wiki/Secure_Hash_Algorithms)

Algoritmus

* [https://en.wikipedia.org/wiki/SHA-2](https://en.wikipedia.org/wiki/SHA-2)

### BCrypt, Argon2

* [https://en.wikipedia.org/wiki/Bcrypt](https://en.wikipedia.org/wiki/Bcrypt)
* [https://en.wikipedia.org/wiki/Argon2](https://en.wikipedia.org/wiki/Argon2)

### Jelszavak tárolása

* [https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
* [https://auth0.com/blog/hashing-passwords-one-way-road-to-security/](https://auth0.com/blog/hashing-passwords-one-way-road-to-security/)
* [https://datatracker.ietf.org/doc/id/draft-whited-kitten-password-storage-02.html](https://datatracker.ietf.org/doc/id/draft-whited-kitten-password-storage-02.html)

## Szimmetrikus kulcsos titkosítás

* Az üzenet kódolásához és dekódolásához is ugyanazt a kulcsot használjuk.
* [https://en.wikipedia.org/wiki/Symmetric-key_algorithm](https://en.wikipedia.org/wiki/Symmetric-key_algorithm)

## Aszimmetrikus kulcsos titkosítás

* A kódoláshoz a publikus, a dekódoláshoz a privát kulcsot használjuk.
* [https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange)

### RSA

* RSA: *Ron Rivest, Adi Shamir, Leonard Adleman*, 1977.

Forrás

* [https://en.wikipedia.org/wiki/RSA_(cryptosystem)](https://en.wikipedia.org/wiki/RSA_(cryptosystem))
* [https://www.practicalnetworking.net/series/cryptography/rsa-example/](https://www.practicalnetworking.net/series/cryptography/rsa-example/)
* [https://crypto.stackexchange.com/questions/1970/how-are-primes-generated-for-rsa](https://crypto.stackexchange.com/questions/1970/how-are-primes-generated-for-rsa)

## Miller-Rabin féle prímteszt

* A Fermat-féle álprímteszthez hasonlóan csak valószínűsíti, hogy az adott egész szám prím-e.
* Különböző egész értékek használhatók benne, így állítható a prímteszt erőssége.
* Lényegesen kisebb a számítási igénye, mint ha végigpróbálgatnánk az összes osztót.
* A gyakorlati felhasználás szempontjából kellő megbízhatóságot ad.

Forrás

* [https://en.wikipedia.org/wiki/Fermat_primality_test](https://en.wikipedia.org/wiki/Fermat_primality_test)
* [https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test)

## Kérdések

* Mi a különbség a hash és az ellenörzőösszeg között?
* Hogyan kapcsolódik Alice és Bob az RSA-hoz?
* Mitől véd, és mitől nem véd a só (*salt*) használata a jelszavak tárolásánál?

<!--
* https://security.stackexchange.com/questions/194600/checksum-vs-hash-differences-and-similarities
* https://stackoverflow.com/questions/460576/hash-code-and-checksum-whats-the-difference
-->

<!-- https://en.wikipedia.org/wiki/Alice_and_Bob -->

## Feladatok

**1. MD5 Hash**

* Ellenőrízzük az algoritmusban szereplő `K` konstans értékeket!
* Vizsgáljuk meg, hogy hogy változnak a `g` értékek az iterációkban!
* Vizsgáljunk meg kész C implementációkat!
* Készítsünk egységteszteket a helyesség ellenörzéséhez!
* Vizsgáljuk meg a bitenkénti forgatás (*left rotate*, *right rotate*) műveletek C-s implementációját!
* Implementáljuk C-ben az algoritmust!
* Készítsük el az OpenCL implementációt (amelyben a hash számítás egy számítási magon végezhető)!

**2. SHA-256**

* Ellenőrízzük az algoritmusban a kiinduló ($h_0, \ldots, h_7$) hash értékeket!
* Ellenőrízzük a `k` konstansok értékét!
* Vizsgáljunk meg kész C implementációkat!
* Készítsünk egységteszteket a helyesség ellenörzéséhez! (Használjuk fel hozzá például az üres sztring hash kódját.)
* Implementáljuk C-ben az algoritmust!

**3. Jelszó visszafejtés**

Egy adatbázisban a `users` táblában a következő adatokat találtuk.

```txt
| name  | password                         |
+-------+----------------------------------+
| user1 | af5b3d17aa1e2ff2a0f83142d692d701 |
| user2 | 3e918e9c9f594bda6b0cf358391c3b1a |
| user3 | a3d11119e7c6af230e2dac2474ef2466 |
| user4 | bf4ab447496f2d3d5a6c77c2cd12f996 |
```

* Az adatbázisban a jelszavak MD5 hash formájában kerültek tárolásra.
* Tudjuk, hogy ASCII kódolású, csak számokat és betűket tartalmazó, pontosan 6 hosszúságú jelszavakról van szó.
* Készítsünk egy OpenCL programot, amelyik segítségével "hatékonyan" visszafejthetjük a jelszavakat!
* Vizsgáljuk meg a futási időt a szekvenciális változathoz képest!
* Adjunk becslést a *jelszó/másodperc* értékére!

**4. Prímtesztelés**

* Implementáljuk C-ben a Miller-Rabin prímtesztelő eljárást (például a determinisztikus változatot)!
* Készítsük el az OpenCL-es változatát!
* Végezzünk méréseket a számítási időre vonatkozóan!

**5. Prímszámok generálása**

* Készítsünk olyan véletlenszám generátort, amelyik $n$ bites prímszámokat tud generálni!
* Az algoritmushoz válasszunk egy nagy, páratlan egész értéket, majd prímteszteléssel viszgáljuk meg, hogy egész-e!

**6. RSA**

* Implementáljuk C-ben a moduláris hatványozást és a Kibővített Euklideszi algoritmust!
* Készítsünk teszteseteket a helyesség ellenörzésére!
* Implementáljuk az RSA algoritmust C-ben!
* Legyen $(99, 19073827)$ a publikus kulcs. Ennek ismeretében próbáljuk meg kitalálni a privát kulcsot!

