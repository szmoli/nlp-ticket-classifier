# Házifeladat beszámoló

## Feladat ismertetése

A házifeladatom célja az volt, hogy egy olyan osztályozó modellt készítsek, ami informatikai problémák szöveges leírásából képes meghatározni viszonylag jó pontossággal, hogy azzal melyik csapatnak kell foglalkoznia, kihez kerüljön a hibajegy.

## Felhasznált adatok

A betanításhoz felhasznált adatokat [innen](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) töltöttem le, de csak az angol nyelvű leírásokat használtam fel (mivel németül nem tudok).

A feladat szempontjából tulajdonképpen csak a `body` és `queue` adatok érdekesek, de a `subject`-et is felhasználtam, hogy kicsit áttekinthetőbb legyen a felhasználói felület. A `body` a probléma szöveges leírása, a `queue` pedig a csapat, aki a hibajeggyel foglalkozik/foglalkozott.

## Applikáció ismertetése

A megoldást három részre osztva készítettem el: adatbázis, frontend és backend.

Az előbb említett három hasznos adatból egy nagyon egyszerű egy-táblás SQLite adatbázist építettem fel. A modellt ebből az adatbázisból tanítom be, illetve az új hibajegyek is itt kerülnek eltárolásra.

A frontend tulajdonképpen csak arra való, hogy az adatbázissal egy kényelmesebb felületen tudjunk interaktálni. A feladat megoldása szempontjából egyáltalán nem fontos, így ehhez a HTML sablonokat a ChatGPT-vel generáltattam le.

A backendhez egy Flask applikáció, amely a HTML sablonokból tényleges HTML-t generál, illetve ez az applikáció futtatja az osztályozó modellt is, mikor egy új hibajegyet adunk fel.

## Osztályozó modell ismertetése

A nyelvfeldolgozó pipeline az alábbi lépésekből áll:

1. Adatok (szöveges leírás, hozzárendelt csapat) lekérdezése az adatbázisból
2. Szöveges leírások tokenizálása (token = szó)
3. Stopword szűrés
4. Szótövesítés
5. Szóbeágyazások készítése a szótövekből
6. Oszályozó modell tanítása a szóbeágyazásokból és csapatokból
7. Csapatok hozzárendelése a szóbeágyazásokhoz.

### Tokenizálás, stopword szűrés és szótövesítés

``` python
def tokenize(text):
    text = re.sub(r'\\[nrtbv]+', ' ', text)  
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)
    return [token.lower() for token in word_tokenize(text)]
```
