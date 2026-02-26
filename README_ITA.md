# Confronto Stima di Densità: Polinomi di Bernstein vs KDE

Questo progetto confronta due metodi per la stima della funzione di densità di probabilità (PDF) partendo da campioni estratti da distribuzioni Erlang:
1. **Polinomi di Bernstein (BPH)**
2. **Kernel Density Estimation (KDE) Standard**

## Struttura dei File

* `bernstein_exp.py`: Contiene le funzioni matematiche core per la creazione della ECDF (Empirical Cumulative Distribution Function) e per il calcolo della PDF basata sui polinomi di Bernstein. **Non deve essere eseguito direttamente.**
* `erlangKK.py`: Script eseguibile per testare coppie specifiche di dimensioni del campione ($M$) e gradi del polinomio ($N$).
* `grid_search.py`: Script eseguibile per effettuare una ricerca a griglia su tutte le combinazioni possibili di $M$ e $N$, utile per l'analisi di sensibilità.

---

## 1. Esecuzione di `erlangKK.py`

Questo script analizza le prestazioni dei due stimatori su coppie mirate di $M$ (numero di campioni) e $N$ (grado del polinomio di Bernstein).

### Configurazioni
All'inizio del file puoi modificare le seguenti variabili:
* `K_VALUES`: Lista dei parametri di forma/scala per le distribuzioni Erlang da testare (es. `[2, 4, 8, 16]`).
* `M_N_PAIRS`: Lista di tuple `(M, N)`. Associa una dimensione del campione a un grado del polinomio specifico.
* `NUM_SIMULATIONS`: Numero di iterazioni indipendenti per ogni configurazione (per avere una stima statistica robusta).
* `NUM_POINTS`: Numero di punti sull'asse x usati per valutare e disegnare le curve.
* `N_PLOT_LINES`: Numero massimo di curve (run) da disegnare effettivamente nei grafici sovrapposti (per evitare grafici troppo pesanti visivamente).
* `GLOBAL_Y_LIM_PDF` / `GLOBAL_Y_LIM_KL`: Limiti fissi per gli assi y dei grafici, utili per confrontare visivamente distribuzioni diverse sulla stessa scala.

### Cosa fa e Risultati Generati
Lo script estrae `M` campioni, calcola le stime BPH e KDE, e misura la Divergenza di Kullback-Leibler (KL) rispetto alla distribuzione Erlang reale.

Al termine, crea una cartella `img/YYYYMMDD_erlangKK/` (dove `YYYYMMDD` è la data odierna) contenente:
1.  **Tabella Riassuntiva (`summary_table_runs{NUM_SIMULATIONS}_erlangKK.jpg`)**: Un'immagine contenente una tabella riepilogativa. Le colonne mostrano la Distribuzione, $M$, $N$, e gli errori medi (KL_bph e KL_KDE) calcolati sulle simulazioni.
2.  **Cartelle delle Distribuzioni (es. `erlang_K2/`)**: All'interno troverai i grafici di confronto (`kde_vs_bernstein_...jpg`). Ogni grafico è una figura 2x2 che mostra:
    * *In alto*: Le curve PDF stimate (in trasparenza) sovrapposte alla ground truth nera (sinistra Bernstein, destra KDE).
    * *In basso*: Boxplot della divergenza KL per mostrare media, mediana e varianza dell'errore.

---

## 2. Esecuzione di `grid_search.py`

Questo script esegue un'analisi esaustiva incrociando tutti i valori forniti per $M$ e $N$. È ideale per capire come il grado del polinomio $N$ influenzi l'errore al variare della quantità di dati $M$.

### Configurazioni
Rispetto al file precedente, qui non si usano coppie fisse, ma due liste separate:
* `M_VALUES`: Lista delle dimensioni del campione (es. `[27, 68, 163, ...]`).
* `N_VALUES`: Lista dei gradi del polinomio di Bernstein (es. `[8, 16, 32, ...]`).
* Le restanti configurazioni (`K_VALUES`, `NUM_SIMULATIONS`, ecc.) mantengono lo stesso significato descritto in `erlangKK.py`. Si aggiunge `GLOBAL_Y_LIM_KL_SENSITIVITY` per gestire l'asse y dei grafici di trend.

### Cosa fa e Risultati Generati
Lo script itera su ogni distribuzione e calcola le metriche per il prodotto cartesiano di `M_VALUES` e `N_VALUES`. 

I risultati vengono salvati nella cartella `img/YYYYMMDD_grid_search/` e comprendono:
1.  **Tabella Riassuntiva (`summary_table_runs{NUM_SIMULATIONS}_grid_search.jpg`)**: Simile a quella del file precedente, ma conterrà molte più righe (una per ogni combinazione di $M$ e $N$ testata).
2.  **Cartelle delle Distribuzioni**: Contengono i grafici di confronto 2x2 per ogni singola combinazione $(M, N)$.
3.  **Cartella `bph_sensitivity/`**: Contiene i grafici di trend (`bph_trend_K{K}.jpg`). 
    * **Come leggere il grafico di trend**: L'asse X rappresenta il grado del polinomio ($N$), mentre l'asse Y (in scala logaritmica) rappresenta l'errore medio KL. Ogni linea colorata rappresenta una diversa dimensione del campione ($M$). Questo grafico è fondamentale per visualizzare visivamente l'esistenza di un eventuale $N$ "ottimo" per un dato $M$, o per osservare il fenomeno dell'overfitting se $N$ cresce troppo.

## Dipendenze Richieste

Versione Python minima: 3.13.0

Per eseguire gli script, assicurati di avere installato:
* `numpy`
* `scipy`
* `matplotlib`


Per il dettaglio delle dipendenze vedere il file `requirements.txt`.
