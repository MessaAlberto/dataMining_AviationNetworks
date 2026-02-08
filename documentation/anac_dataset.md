### **Dataset Report: ANAC Brazil - Voo Regular Ativo (VRA)**

**Fonte Dati e Documentazione**

* **Nome Dataset:** Voo Regular Ativo (VRA)
* **Fonte:** Agência Nacional de Aviação Civil (ANAC) - Governo del Brasile
* **Link Ufficiale / Dizionario Dati:** [Voo Regular Ativo (VRA) - ANAC](https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/voos-e-operacoes-aereas/voo-regular-ativo-vra/62-voo-regular-ativo-vra)

**Descrizione Generale**
Questo dataset contiene lo storico delle operazioni aeree in Brasile. A differenza del BTS americano che fornisce metriche pre-calcolate, il VRA fornisce i dati grezzi ("raw timestamps") degli orari previsti ed effettivi. Questo richiede un passaggio di *feature engineering* per calcolare i ritardi.

---

#### **1. Campi Utilizzati per la Topologia (Costruzione del Grafo)**

Questi dati definiscono i nodi e gli archi della rete sudamericana.

* **`ICAO Aeródromo Origem`**: Codice ICAO (4 lettere) dell'aeroporto di partenza (es. "SBGL").
* *Utilizzo:* **Nodo Sorgente**.
* *Attenzione:* Il grafo globale usa codici **IATA** (3 lettere). È necessaria una conversione (es. `SBGL` -> `GIG`) usando il dizionario `airports.csv`.


* **`ICAO Aeródromo Destino`**: Codice ICAO dell'aeroporto di arrivo.
* *Utilizzo:* **Nodo Destinazione**. Definisce l'arco diretto.



#### **2. Campi Utilizzati per la Performance (Calcolo Fragilità)**

Questi dati sono necessari per derivare la metrica di inefficienza.

* **`Partida Prevista`**: Data e ora schedulata della partenza (formato timestamp).
* *Utilizzo:* Baseline per calcolare il ritardo.


* **`Partida Real`**: Data e ora effettiva in cui l'aereo ha lasciato il gate (o decollato, a seconda della definizione ANAC, solitamente "off-block").
* *Utilizzo:* Confrontato con la `Prevista`, genera il ritardo.


* **`Situação Voo`**: Stato del volo (es. "REALIZADO", "CANCELADO").
* *Utilizzo:* **Filtro critico**. Dobbiamo escludere voli cancellati o non operati perché i loro campi orari potrebbero essere nulli o non validi.



---

### **Pipeline di Trasformazione (Data Engineering)**

Poiché il dataset non ha la colonna "Ritardo in Minuti", ecco la logica obbligatoria da documentare nel report:

1. **Filtering:**
* `WHERE "Situação Voo" == 'REALIZADO'`
* *Motivo:* Garantisce che stiamo analizzando solo voli effettivamente partiti.


2. **Calculation (Delay):**
* Formula: `(Partida Real - Partida Prevista)` in minuti.
* *Nota:* Questo genera valori negativi per gli anticipi.


3. **Normalization (Coerenza con BTS):**
* Formula Applicata: `max(0, Delay)`
* *Motivo:* Per allinearsi alla logica USA (`DepDelayMinutes`), gli anticipi vengono portati a 0. Se un volo parte 10 minuti prima, non "cancella" il ritardo di un altro volo. L'inefficienza non può essere negativa.


4. **Mapping (Standardizzazione):**
* Conversione: `ICAO`  `IATA`.
* *Motivo:* Per permettere il `merge` con il grafo globale costruito su OpenFlights (che usa IATA).



---

### **Dizionario Variabili (Sintesi per il Codice)**

| Nome Colonna CSV | Tipo Dato | Ruolo nel Progetto | Azione Richiesta |
| --- | --- | --- | --- |
| `ICAO Aeródromo Origem` | String (4 char) | Nodo Source | Convertire in IATA |
| `Partida Prevista` | DateTime | Reference Time | Parsing data |
| `Partida Real` | DateTime | Actual Time | Parsing data + Calcolo Delta |
| `Situação Voo` | String | Filtro Qualità | Tenere solo "REALIZADO" |
| `Código Justificativa` | String | Analisi Cause | (Opzionale) Utile per capire cause esterne |