# 📈 SurvivaLab

> Carica un CSV. Scegli le colonne. Ottieni curve di Kaplan-Meier pronte da pubblicare.

App Streamlit per survival analysis interattiva — data quality check, curve KM stratificate, log-rank test, download PNG alta risoluzione.

Puoi visitarlo qui 👉🏻 [SurvivaLab](https://ricciarello-survivalab.streamlit.app/)

## Funzionalità

- **Data Quality Check** automatico (duplicati, mancanti, valori negativi, validità EVENT)
- **Curva KM singola** o **stratificata** per 1+ variabili categoriche
- **Log-rank test** automatico (2 gruppi: test standard; 3+ gruppi: test multivariato)
- **Opzioni grafiche**: banda CI 95%, tick censure, linea mediana, tabella at-risk
- **Download PNG** alta risoluzione (180 dpi)

## Setup locale

```bash
git clone https://github.com/ricciarello/survivalab
cd survivalab
pip install -r requirements.txt
streamlit run app.py
```

## Formato CSV atteso

| Colonna | Tipo | Descrizione |
|---|---|---|
| TIME | numerico | Durata osservazione |
| EVENT | 0/1 | 1 = evento avvenuto, 0 = censurato |
| (opzionale) | qualsiasi | Variabili di stratificazione |

Usa `sample_data.csv` per testare.

## Stack

- `lifelines` — Kaplan-Meier, log-rank test
- `matplotlib` — visualizzazione
- `streamlit` — deployment

---

*Progetto portfolio · [ricciarello](https://github.com/ricciarello)*
