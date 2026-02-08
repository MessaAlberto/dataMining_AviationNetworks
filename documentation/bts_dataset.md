### **Dataset Report: US Airline On-Time Performance (BTS)**

**Descrizione Generale**
Il dataset proviene dal database *TranStats* del Dipartimento dei Trasporti USA (DOT) e contiene i dati sulle prestazioni di puntualità dei vettori aerei domestici. Ogni riga rappresenta un singolo volo commerciale, dettagliando orari pianificati ed effettivi, ritardi e cause specifiche.

#### **1. Campi Utilizzati per la Topologia (Costruzione del Grafo)**

Questi dati definiscono la struttura della rete aerea .

* **`Origin` (Origin Airport Code):** Il codice IATA dell'aeroporto di partenza (es. "JFK").
* *Utilizzo:* Rappresenta il **Nodo Sorgente** nel grafo. È la chiave primaria per raggruppare i dati e calcolare le metriche di "fragilità" (ritardo medio in uscita).


* **`Dest` (Destination Airport Code):** Il codice IATA dell'aeroporto di arrivo.
* *Utilizzo:* Rappresenta il **Nodo Destinazione**. Definisce l'esistenza di un arco diretto tra due nodi.


* **`FlightDate` / `Year` / `Month`:** Riferimenti temporali del volo.
* *Utilizzo:* Necessari per filtrare il dataset su un periodo specifico (es. Anno 2024) e garantire la coerenza temporale dell'analisi.



#### **2. Campi Utilizzati per la Performance (Analisi di Fragilità)**

Questi dati quantificano l'efficienza operativa dei nodi.

* **`DepDelayMinutes` (Departure Delay Minutes):** La differenza in minuti tra l'orario di partenza effettivo e quello schedulato.
* *Nota Critica:* A differenza del campo `DepDelay` grezzo, qui i valori negativi (partenze in anticipo) sono impostati a **0**.
* *Perché lo usiamo:* È la metrica "pura" dell'inefficienza. Ci interessa sapere "quanto tempo si perde", non se un volo ha recuperato tempo. Evita che i voli in anticipo abbassino artificialmente la media dei ritardi gravi.


* **`Cancelled`:** Indicatore binario (1 = Cancellato).
* *Perché lo usiamo:* Per la pulizia dei dati. I voli cancellati non hanno un tempo di ritardo numerico valido e vanno esclusi dal calcolo della media (`DepDelayMinutes`), oppure trattati separatamente come fallimento totale del nodo.


* **`Diverted`:** Indicatore binario (1 = Dirottato).
* *Perché lo usiamo:* Simile ai cancellati, questi voli non completano la rotta prevista e vanno filtrati per non sporcare le metriche di ritardo standard.



#### **3. Campi Avanzati (Analisi delle Cause - Opzionale ma Consigliato)**

Questi campi permettono di capire *perché* un Hub è un collo di bottiglia.

* **`NASDelay` (National Air System Delay):** Ritardi in minuti attribuiti al sistema aeroportuale (es. traffico intenso, controllo traffico aereo).
* *Perché lo usiamo:* È la "prova del nove". Se un aeroporto ha un alto `Avg_Delay` correlato a un alto `NASDelay`, significa che è **strutturalmente saturo** (un vero collo di bottiglia fisico).


* **`CarrierDelay` / `LateAircraftDelay`:** Ritardi imputabili alla compagnia aerea o all'aereo in ritardo dal volo precedente.
* *Perché lo usiamo:* Per distinguere i problemi dell'aeroporto dai problemi logistici delle compagnie aeree.



---

### **Sintesi della Logica di Estrazione**

| Variabile Progetto | Campo Dataset BTS | Motivazione |
| --- | --- | --- |
| **Nodo (Source)** | `Origin` | Identifica l'aeroporto che stiamo valutando (chi genera il ritardo?). |
| **Arco (Edge)** | `Dest` | Definisce la connettività per il calcolo di PageRank/Betweenness. |
| **Metric: Inefficiency** | `DepDelayMinutes` | Misura il tempo perso alla partenza (clamped a 0 per evitare "bonus" da anticipi). |
| **Filter: Valid Flights** | `Cancelled == 0` | Escludiamo i voli non partiti per avere una media ritardo pulita. |