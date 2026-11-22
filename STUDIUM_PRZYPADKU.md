# Studium przypadku: system wspomagania decyzji logistycznych

## 1. Cel i zakres projektu

Projekt przedstawia system wspomagania decyzji logistycznych, który analizuje
parametry przesyłki i przygotowuje rekomendację dla pracownika logistyki.
System łączy predykcję czasu dostawy, deterministyczną ocenę ryzyka, retrieval
augmented generation (RAG), cztery role LLM oraz reguły biznesowe zapisane w
Pythonie.

Projekt ma charakter portfolio i studium przypadku. Nie jest deklarowany jako
pełny system produkcyjny: wyceny przewoźników pochodzą z lokalnej,
wersjonowanej karty stawek, a nie z rzeczywistego API przewoźnika.

### Użytkownicy

- pracownik działu logistyki analizujący przesyłki,
- operator obsługi klienta przygotowujący komunikat o opóźnieniu,
- menedżer transportu porównujący przewoźników i koszty,
- osoba oceniająca jakość rekomendacji systemu.

### Podstawowy sposób działania

1. FastAPI przyjmuje i waliduje żądanie.
2. Jeżeli czas dostawy nie został podany, model XGBoost wyznacza predykcję.
3. ChromaDB wyszukuje kontekst z lokalnej bazy wiedzy logistycznej.
4. Python oblicza autorytatywny wynik i poziom ryzyka.
5. Agent ryzyka przygotowuje wyjaśnienie, ale nie może zmienić wyniku
   obliczonego przez Python.
6. Agent przewoźnika korzysta z typowanych, deterministycznych wycen.
7. Agent strategii odzyskania przygotowuje komunikację i ewentualny voucher.
8. Dwaj specjaliści działają równolegle, a Agent 4 integruje ich wyniki.
9. Pydantic waliduje decyzję, a `ResponseGrader` ocenia jakość odpowiedzi.
10. Dla `/analyze` wynik może zostać zapisany w audycie PostgreSQL. Redis
    pełni funkcję opcjonalnego cache i nie jest wymagany do poprawności analizy.

Najważniejsza decyzja projektowa polega na rozdzieleniu odpowiedzialności:
LLM generuje wyjaśnienia i rekomendacje, natomiast Python kontroluje wynik
ryzyka, poziom ryzyka, ceny, politykę voucherów i końcową normalizację.

## 2. Wymagania jakościowe

| ID | Wymaganie | Sposób realizacji |
|---|---|---|
| Q1 | Poprawność reguł biznesowych | Wynik ryzyka, ceny przewoźników i polityka voucherów są wyliczane lub weryfikowane deterministycznie. |
| Q2 | Odporność na błędne odpowiedzi LLM | Odpowiedzi są walidowane schematami Pydantic i normalizowane przed użyciem. |
| Q3 | Wyjaśnialność | Odpowiedź zawiera poziom ryzyka, wynik punktowy, czynniki ryzyka, rekomendację i grading. |
| Q4 | Niezawodność | Timeouty, ograniczone retry oraz jawna obsługa błędu dostawcy LLM ograniczają niekontrolowane zawieszenie. |
| Q5 | Degradacja kontrolowana | Awaria Redis nie blokuje analizy, a fallback deterministyczny jest jawnie przeznaczony dla publicznego demo. |
| Q6 | Bezpieczeństwo | Analiza uwierzytelniona wymaga `x-api-key`, a sekrety są dostarczane przez zmienne środowiskowe. |
| Q7 | Testowalność | Wywołania LLM są mockowane, a reguły i integracja API mają testy automatyczne. |
| Q8 | Wydajność | Przygotowanie blokujące jest wykonywane poza pętlą FastAPI, a agenci przewoźnika i odzyskania działają równolegle. |
| Q9 | Utrzymywalność | Orkiestracja jest jawna, a przepływ systemu opisano diagramami i dokumentacją architektury. |

## 3. Kryteria akceptacji

Kryterium uznaję za spełnione, gdy poniższy warunek jest prawdziwy:

| ID | Mierzalny warunek | Status |
|---|---|---|
| A1 | Żądanie `/analyze` bez poprawnego klucza zwraca `401` lub `403`. | Spełnione, test API |
| A2 | Żądanie z brakującymi wymaganymi polami zwraca `422`. | Spełnione, test API |
| A3 | Przekroczenie limitu analizy zwraca `504`. | Spełnione, test API |
| A4 | Błąd dostawcy LLM na `/analyze` zwraca `503`, zamiast udawać poprawny wynik. | Spełnione, test API |
| A5 | Dla identycznego wejścia narzędzie wyceny zwraca identyczny, typowany wynik. | Spełnione, test narzędzia |
| A6 | Wynik ryzyka jest zgodny z regułami punktowymi i odpowiednim przedziałem poziomu. | Spełnione, testy regresji |
| A7 | Cena wygenerowana przez LLM nie może zastąpić ceny z karty stawek. | Spełnione, test narzędzia |
| A8 | Voucher i procent rabatu są zgodne z polityką odzyskania. | Spełnione, testy agenta i gradingu |
| A9 | Poprawna analiza zawiera decyzję oraz grading ryzyka, przewoźnika i odzyskania. | Spełnione, test integracyjny |
| A10 | Pełna suite testów kończy się bez niepowodzeń. | Spełnione: 60/60 testów |

Kryteria dotyczące maksymalnej latencji produkcyjnej, rzeczywistego API
przewoźnika i limitowania ruchu nie są jeszcze kryteriami spełnionymi. Są
propozycjami dla kolejnej wersji, ponieważ lokalne środowisko testowe nie
zapewnia reprezentatywnego obciążenia produkcyjnego.

## 4. Rejestr ryzyk

Ocena: prawdopodobieństwo i skutek są określane w skali Niskie/Średnie/Wysokie.
Priorytet wynika z połączenia obu wartości i znaczenia dla użytkownika.

| Ryzyko | Prawdopodobieństwo | Skutek | Priorytet | Postępowanie |
|---|---|---|---|---|
| LLM zwróci błędną lub sprzeczną rekomendację | Średnie | Wysoki | Wysoki | Schematy Pydantic, normalizacja i deterministyczne reguły nadrzędne nad LLM. |
| Dostawca LLM będzie niedostępny lub przekroczy limit | Średnie | Wysoki | Wysoki | Timeout, ograniczone retry i jawny błąd `503` dla uwierzytelnionej analizy. |
| Predykcja XGBoost będzie niedokładna dla nowego przypadku | Średnie | Wysoki | Wysoki | Monitorowanie jakości modelu, ponowna walidacja danych i przyszłe testy na zbiorze referencyjnym. |
| Lokalna karta stawek będzie nieaktualna | Średnie | Średni | Średni | Wersjonowanie rate card i docelowo integracja z autoryzowanym API przewoźnika. |
| Kontekst RAG będzie nieaktualny lub nieadekwatny | Średnie | Średni | Średni | Kontrolowana baza dokumentów i jawna procedura ponownego indeksowania. |
| Redis będzie niedostępny | Średnie | Niski | Niski | Pominięcie cache; Redis jest optymalizacją, a nie zależnością poprawności. |
| PostgreSQL będzie niedostępny | Średnie | Średni | Niski | Zapis audytowy jest best-effort i nie powinien odrzucać poprawnej analizy. |
| Publiczna usługa zostanie przeciążona | Średnie | Wysoki | Wysoki | Obecnie ryzyko otwarte: `rate_limit_check` jest tylko punktem rozszerzenia; docelowo należy użyć nginx lub API gateway. |
| Równoległość i timeout zwiększą złożoność diagnostyki | Niskie | Średni | Niski | Jawna orkiestracja, testy integracyjne i logowanie etapów. |

Najwyższy priorytet mają zależność od dostawcy LLM, jakość predykcji oraz brak
produkcyjnego rate limitingu. Są to obszary, które mogą wpłynąć na decyzję
biznesową albo dostępność usługi.

## 5. Plan testów

### Zakres

Testowane są:

- walidacja danych wejściowych i autoryzacja endpointów,
- obliczanie i normalizacja ryzyka,
- typowane wyceny przewoźników,
- polityka voucherów i komunikacji,
- odporność na błędne dane zwrócone przez LLM,
- grading odpowiedzi,
- timeouty i błędy dostawcy,
- pełny przepływ API z zamockowanym LLM.

### Rodzaje testów

1. **Testy jednostkowe** — sprawdzają pojedyncze funkcje, m.in. reguły ryzyka,
   grading, parser JSON i kartę stawek.
2. **Testy integracyjne API** — uruchamiają endpointy FastAPI z użyciem
   `TestClient` i sprawdzają walidację, autoryzację oraz strukturę odpowiedzi.
3. **Testy regresyjne** — chronią zasadę, że LLM nie może zmienić wyniku
   ryzyka, ceny z rate card ani polityki voucherów.
4. **Testy walidacyjne i negatywne** — sprawdzają brak klucza API, brak pól,
   pusty batch, niepoprawny JSON i błędny scenariusz.
5. **Testy odporności** — symulują timeout, niedostępność LLM, retry oraz
   częściową porażkę batcha.
6. **Testy kontraktów i typów** — sprawdzają modele Pydantic oraz typowany
   wynik narzędzia przewoźnika.
7. **Testy health-check i stanu aplikacji** — sprawdzają `/health`, `/status`
   oraz podstawowe informacje o gotowości komponentów.

Testy wydajnościowe pod dużym obciążeniem, testy penetracyjne oraz testy
rzeczywistego wdrożenia z zewnętrznym LLM nie są częścią obecnej automatycznej
suite. Wymagałyby osobnego środowiska i nie powinny korzystać z prawdziwego
limitu dostawcy LLM.

### Przykładowe scenariusze

| ID | Rodzaj | Dane wejściowe / działanie | Oczekiwany wynik |
|---|---|---|---|
| T01 | Negatywny API | `POST /analyze` bez `x-api-key`. | Status `401` lub `403`; analiza nie jest uruchamiana. |
| T02 | Walidacyjny API | `POST /analyze` bez wymaganych pól przesyłki. | Status `422`; FastAPI odrzuca dane wejściowe. |
| T03 | Jednostkowy, graniczny | Dystans `1500 km`, następnie `1501 km`. | Odpowiednio `0` i `+20` punktów. |
| T04 | Jednostkowy, graniczny | Waga `3000 g`, następnie `3001 g`. | Odpowiednio `0` i `+15` punktów. |
| T05 | Jednostkowy, graniczny | Opóźnienie `3 dni`, następnie `3.1 dnia`. | Punkty są dodane dopiero powyżej `3 dni`. |
| T06 | Regresyjny LLM | Mock LLM podaje koszt `999999`. | Zwrócony koszt pochodzi z rate card. |
| T07 | Regresyjny LLM | Mock LLM podaje voucher bez opóźnienia. | Voucher zostaje usunięty, a rabat wynosi `0`. |
| T08 | Jednostkowy, graniczny | Wynik `80`, następnie `81`. | Odpowiednio poziom `HIGH` i `CRITICAL`. |
| T09 | Odpornościowy | Symulowany timeout pełnej analizy. | Endpoint `/analyze` zwraca `504`. |
| T10 | Odpornościowy | Dostawca LLM zgłasza błąd na `/analyze`. | Endpoint zwraca `503`; błąd nie jest maskowany fallbackiem. |
| T11 | Odpornościowy batch | Pierwszy element poprawny, drugi kończy się `LLMError`. | `completed_with_errors`, `successful == 1`, `failed == 1`; wynik pierwszego zostaje zachowany. |
| T12 | Negatywny batch | `POST /batch-analyze` z pustą listą. | Status `400` i komunikat `Empty request list`. |
| T13 | Grading | Niepoprawny JSON, np. `not valid json {{{`. | Grading wynosi `0` i zawiera informację o błędzie. |
| T14 | Deterministyczny | Dwukrotne wywołanie tej samej wyceny przewoźnika. | Wyniki są identyczne i mają źródło `portfolio_rate_card_v1`. |
| T15 | Integracyjny demo | `POST /demo/analyze` dla `high` bez API key. | Publiczny endpoint zwraca decyzję i grading. |
| T16 | Cache/fallback | Powtórzenie demo albo brak dostawcy LLM. | Odpowiednio `cache_hit == true` albo jawne `fallback_used == true`. |

Testy LLM nie korzystają z prawdziwego dostawcy ani z jego limitu. Odpowiedzi
agenta są mockowane, dzięki czemu wynik testów jest powtarzalny.

## 6. Realizacja i wyniki testów

Testy uruchomiono w środowisku:

- Python 3.12,
- lokalne środowisko `.venv`,
- polecenie: `python -m pytest -q`.

Wynik wykonania:

```text
60 passed, 2 warnings in 7.81s
```

Nie wystąpiły niepowodzenia testów. Wystąpiły dwa ostrzeżenia:

1. ostrzeżenie biblioteki `httpx` o przestarzałym skróconym sposobie tworzenia
   klienta testowego,
2. ostrzeżenie XGBoost dotyczące ładowania serializowanego modelu utworzonego
   w starszej wersji.

Ostrzeżenia nie spowodowały niepowodzenia testów, ale powinny zostać wpisane
do zadań utrzymaniowych. Wynik nie jest deklaracją gotowości produkcyjnej ani
wynikiem testu wydajnościowego pod dużym obciążeniem.

## 7. Wnioski

System spełnia przyjęte wymagania jakościowe w zakresie poprawności
deterministycznych reguł, walidacji danych, odporności na niepoprawne
odpowiedzi LLM, testowalności i podstawowego bezpieczeństwa. Najważniejsze
wartości biznesowe nie są pozostawione wyłącznie modelowi językowemu.

Najbardziej ryzykowne obszary to:

- dostępność i jakość odpowiedzi zewnętrznego dostawcy LLM,
- dokładność predykcji czasu dostawy dla przypadków innych niż dane treningowe,
- aktualność lokalnej karty stawek i dokumentów RAG,
- brak rzeczywistego limitowania ruchu na poziomie aplikacji lub bramy API.

Najbardziej wartościowe kolejne usprawnienia to:

1. testy wydajnościowe i pomiar p95/p99 czasu odpowiedzi,
2. monitoring jakości predykcji XGBoost po wdrożeniu,
3. integracja z rzeczywistymi, uwierzytelnionymi API przewoźników,
4. centralny rate limiting i monitoring bezpieczeństwa,
5. okresowa ocena jakości kontekstu RAG i aktualizacja dokumentów,
6. testy kontraktowe dla zewnętrznego dostawcy LLM.

Projekt jest więc odpowiednim przykładem implementacyjnego studium przypadku:
posiada konkretny cel, mierzalne kryteria akceptacji, jawne ryzyka, plan
testów, działającą implementację oraz powtarzalne wyniki testów.
