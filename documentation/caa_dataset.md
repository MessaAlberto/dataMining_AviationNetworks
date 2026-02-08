### **Dataset Report: UK Flight Punctuality Statistics (CAA)**

**Fonte Dati e Documentazione**

* **Nome Dataset:** Annual Punctuality Statistics Full Analysis Arrival Departure
* **Fonte:** UK Civil Aviation Authority (CAA)
* **Link Definizioni:** [CAA Punctuality Notes & Definitions](https://www.caa.co.uk/data-and-analysis/uk-aviation-market/flight-punctuality/notes/)

**Descrizione Generale**
Questo dataset fornisce statistiche aggregate (mensili o annuali) per rotta. A differenza dei file USA e Brasile che sono "per singolo volo" (row-by-flight), qui ogni riga è già un **riassunto** di tutti i voli operati da una specifica compagnia aerea su una specifica rotta in quel periodo.

---

#### **1. Campi Utilizzati per la Topologia (Costruzione del Grafo)**

Questi dati definiscono i nodi e gli archi, ma richiedono una pulizia specifica per i nomi.

* **`reporting_airport`**: Il nome dell'aeroporto UK che riporta i dati (es. "ABERDEEN", "HEATHROW").
* *Utilizzo:* **Nodo Sorgente** (quando filtriamo per partenze).
* *Attenzione:* È un **Nome Completo**, non un codice IATA. Il tuo script `plot.py` dovrà usare un dizionario per convertirlo (es. `HEATHROW`  `LHR`).


* **`origin_destination`**: Il nome della destinazione (es. "TORONTO").
* *Utilizzo:* **Nodo Destinazione**.
* *Sfida:* Spesso indica la città e non l'aeroporto specifico (es. "TORONTO" potrebbe essere YYZ o YTZ). Per un'analisi macroscopica, mappalo all'aeroporto principale della città.



#### **2. Campi Utilizzati per la Performance (Metrica di Fragilità)**

Questi sono i numeri che userai per calcolare l'efficienza.

* **`average_delay_mins`**: Il ritardo medio in minuti.
* *Definizione:* Differenza tra "Actual Gate Time" (Off-Block) e orario schedulato. Include tutti i ritardi.
* *Nota:* Se la casella è vuota (come nel tuo esempio), significa che non ci sono stati voli validi.


* **`number_flights_matched`**: Numero di voli operati e abbinati a uno slot.
* *Utilizzo:* **Peso e Filtro**.
* *Importante:* Nel tuo esempio la riga ha valore `0`. Queste righe vanno scartate immediatamente (nessun volo = nessun dato di ritardo).


* **`arrival_departure`**: Indicatore di direzione.
* *Utilizzo:* **Filtro Fondamentale**. Devi tenere solo le righe con valore **`D`** (Departure) per misurare il ritardo generato dall'aeroporto UK.



---

### **Pipeline di Trasformazione (Data Engineering)**

Poiché questo file è aggregato e usa nomi invece di codici, ecco la logica da seguire nel codice:

1. **Filtro Direzione:**
* `WHERE arrival_departure == 'D'`
* *Motivo:* Vogliamo analizzare quanto l'aeroporto `reporting_airport` ritarda i voli in uscita.


2. **Filtro Validità:**
* `WHERE number_flights_matched > 0`
* *Motivo:* Come vedi nella tua riga di esempio (Aberdeen-Toronto), ci sono 0 voli e il ritardo è vuoto. Queste righe sono rumore statistico (rotte pianificate ma non operate).


3. **Calcolo della Metrica Pesata (Opzionale ma Preciso):**
* Poiché ogni riga è già una media, se vuoi aggregare più compagnie sulla stessa rotta, non puoi fare la media delle medie.
* *Formula Corretta:* 


4. **Mapping Geografico (Il passo più delicato):**
* Devi convertire i Nomi in IATA.
* *Esempio:* Crea una mappa nel codice:
```python
uk_map = {
    "ABERDEEN": "ABZ",
    "HEATHROW": "LHR",
    "GATWICK": "LGW",
    ...
}

```


* Senza questo passaggio, non potrai collegare questi dati al Grafo Globale (che parla solo lingua IATA).



---

### **Dizionario Variabili (Sintesi per il Codice)**

| Nome Colonna CSV | Tipo Dato | Ruolo nel Progetto | Azione Richiesta |
| --- | --- | --- | --- |
| `reporting_airport` | String | Nodo Source | Convertire Nome  IATA |
| `origin_destination` | String | Nodo Target | Convertire Nome  IATA (best effort) |
| `arrival_departure` | String ('A'/'D') | Filtro Logico | Tenere solo **'D'** |
| `average_delay_mins` | Float | Metrica Inefficienza | Usare tal quale (handle NaN) |
| `number_flights_matched` | Integer | Peso Statistico | Scartare se **0** |