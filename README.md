# 🤖 AgenticAI - Multi-Agent Logistics Decision System

System inteligencji decyzyjnej łączący predykcje ML z architekturą Multi-Agent LLM do autonomicznej analizy ryzyka opóźnień dostaw e-commerce.

---

## 🎯 Co robi ten system?

System przyjmuje dane o zamówieniu (waga, dystans, opóźnienie płatności, itp.) i:

1. **Predykcja ML** - Model XGBoost przewiduje czas dostawy
2. **Kontekst RAG** - ChromaDB wyszukuje relevantne zasady logistyczne
3. **Analiza Agentów** - 4 wyspecjalizowane agenty AI analizują scenariusz równolegle
4. **Decyzja** - System generuje rekomendacje: voucher, zmiana przewoźnika, komunikacja z klientem

---

## 🧠 Architektura Multi-Agent

System wykorzystuje 4 wyspecjalizowane agenty LLM (Mistral 7B via Ollama):

| Agent | Odpowiedzialność |
|-------|------------------|
| **Risk Agent** | Ocena ryzyka (0-100), identyfikacja czynników ryzyka |
| **Carrier Agent** | Rekomendacja przewoźnika, analiza ROI upgrade'u |
| **Recovery Agent** | Strategia retencji klienta, kody voucherów |
| **Orchestrator** | Synteza wszystkich analiz, podsumowanie wykonawcze |

---

## 🛠️ Tech Stack

- **Python 3.10+** - Język programowania
- **Ollama + Mistral 7B** - Lokalny LLM (bez API keys)
- **ChromaDB** - Baza wektorowa dla RAG
- **XGBoost** - Model predykcji czasu dostawy
- **Pydantic** - Walidacja typów odpowiedzi LLM
- **Redis** - Cache dla przyspieszenia (opcjonalnie)
- **Rich** - Interfejs CLI

---

## 📦 Instalacja

```bash
# Klonuj repozytorium
git clone https://github.com/GabrielWalak/Agentic-Logistics-Optimizer.git
cd Agentic-Logistics-Optimizer

# Stwórz środowisko wirtualne
python -m venv .venv
.venv\Scripts\activate  # Windows

# Zainstaluj zależności
pip install pydantic ollama chromadb sentence-transformers xgboost scikit-learn pandas rich python-dotenv

# Pobierz model Mistral dla Ollama
ollama pull mistral

# Uruchom system
python main.py
```

---

## 🎮 Użycie

### Uruchomienie pełnego systemu
```bash
python main.py
```

### Test połączenia z agentami
```bash
python pydantic_agents.py
```

---

## 📸 Screenshots

### System startup i inicjalizacja
![System Initialization](screenshots/1.png)

### Analiza scenariusza przez Multi-Agent System
![Agent Analysis](screenshots/2.png)

### Wynik końcowy z rekomendacjami
![Final Decision](screenshots/3.png)

---

## 📁 Struktura projektu

```
AgenticAI/
├── main.py                     # Punkt wejścia - orkiestracja workflow
├── pydantic_agents.py          # System multi-agentowy z modelami Pydantic
├── chroma_db_manager.py        # Manager bazy wektorowej ChromaDB
├── logistics_knowledge_base.py # Dokumenty domenowe dla RAG
├── logistics_docs/             # Pliki źródłowe wiedzy
│   ├── carrier_rules.txt
│   ├── customer_recovery.txt
│   ├── distance_guidelines.txt
│   └── ...
└── screenshots/                # Screenshoty z aplikacji
```

---

## 🔧 Jak to działa?

### 1. Input
```python
scenario = {
    'product_weight_g': 5000,      # Ciężka paczka
    'distance_km': 1200.0,         # Długi dystans
    'payment_lag_days': 2,         # Opóźnienie płatności
    'is_weekend_order': 1,         # Zamówienie weekendowe
    'freight_value': 85.00         # Koszt frachtu
}
```

### 2. XGBoost Prediction
Model przewiduje: **9.2 dni** (obiecano 7 dni → RYZYKO OPÓŹNIENIA)

### 3. RAG Context
ChromaDB znajduje relevantne dokumenty:
- "Dystans >800km wymaga Premium Express"
- "Weekend orders +1-2 dni processing"

### 4. Multi-Agent Analysis
Agenty analizują równolegle i zwracają:
- **Risk Score**: 85/100 (HIGH)
- **Carrier**: Upgrade do Premium Express
- **Voucher**: DELAY50 (50% zniżki na kolejne zamówienie)
- **Confidence**: 90/100

---

## 📊 Wydajność

| Metryka | Wartość |
|---------|---------|
| Czas analizy (bez cache) | 30-45 sekund |
| Czas analizy (z cache Redis) | ~2 sekundy |
| Model XGBoost R² | 0.41 |

---

## 📄 Licencja

MIT License

---

<p align="center">
  <strong>Built with 🤖 Ollama + ChromaDB + XGBoost</strong>
</p>
