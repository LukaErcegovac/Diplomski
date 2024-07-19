# Diplomski

Implementacija metoda je odrađena tri puta: samostalnom implementacijom koraka svake metode, korištenjem biblioteka i samostalnom implementacijom koraka, ali bez dodavanja šuma na sliku kako bi se metoda mogla primjeniti na slikama gdje šum već postoji. Tijekom implementacije korišteno je virtualno okruženje (venv) i sve korištene biblioteke spremljene su u dokuent "requirements.txt".

Sve je strukturirano na načina da svaki dokument sa imenom metode u sebi sadrži  sva tri načina implementacije, a to izgleda ovako:

### mean-filter:
  - mean-filter-librery.py --> implementacija metode lokalnog usrednjavanja korištenjem biblioteke, korištena zbog testiranja samostalne implementacije.
  - mean-filter-without-adding-noise.py --> samostalna implementacija koraka metode lokalnog usrednjavanja bez dodavanja šuma, koristi se za slike gdje šum već postoji.
  - mean-filter.py --> samostalna implementacija koraka metoda lokalnog usrednjavanja.

### non-local-mean-filter:
  - non-local-mean-filter-librery.py --> implementacija metode nelokalnog usrednjavanja korištenjem biblioteke, korištena zbog testiranja samostalne implementacije.
  - non-local-mean-filter-without-adding-noise.py --> samostalna implementacija koraka metode nelokalnog usrednjavanja bez dodavanja šuma, koristi se za slike gdje šum već postoji.
  - non-local-mean-filter.py --> samostalna implementacija koraka metode nelokalnog usrednjavanja.

### wavelet-filter:
  - wavelet-filter-librery.py --> implementacija metode korištenja valića korištenjem biblioteke, korištena zbog testiranja samostalne implementacije.
  - wavelet-filter-without-adding-noise.py --> samostalna implementacija koraka metode korištenja valića bez dodavanja šuma, koristi se za slike gdje šum već postoji.
  - wavelet-filter.py --> samostalna implementacija koraka metode korištenja valića.

## Korištenje i pokretanja koda

Ako se želi koristiti virtualno okruženje potrebno ga je instalirati. Virtualno okruženje može se koristit kako instalirane biblioteke ne bi bile instalirane lokalno i jednom kada se dokument izbrše i one će biti izbrisane.

Instaliranje virtualnog okruženja:
  - pip install virtualenv
  - python -m venv \<ime virtualnog okruženja>

Virtulno okruženje može se pokrenuti ovako:
  - cd .\venv\Scripts\
  - .\activate

Ako se pojavljuje greška onda je prije pokretanja druge naredbe (".\activate") potrebno pokrenuti sljedeću naredbu:
  - Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Kada se pokrene virutalno okruženje potrebno je vratiti se natrag u glavni dokument, a to se može postići naredbom:
 - cd .. --> naredb je potrebno pokrenuti dva puta

Bitno je da se instaliraju sve potrebne biblioteke za rad, a one se nalaze u "requirements.txt":
  - pip install requirements.txt

Nakon što su instalirane sve potrebne biblioteke može se odabrati koju metodu se želi koristiti, a kao što je prije objašnjeno implementacija svake metode je u zasebnom dokumentu.
Prvo je potrebno da se uđe u dokument metode koja se želi koristit, a to se može napraviti pomoću naredbe "cd":
  - cd \<ime dokumenta>

Nakon što je odabran dokument python dokumenti u njemu se mogu pokrenuti pomoću sljedeće naredbe:
  - python \<ime python dokumenta>.py
