# Data Mining Project: Data-Driven Analysis and Recommendation in the Global Aviation Network

This project investigates the discrepancy between topological importance and operational efficiency within the global aviation network. It challenges the "Fragility Paradox"—the hypothesis that central hubs are systemic bottlenecks—by integrating static network topology (OpenFlights) with millions of real-world flight records from the US, UK, and Brazil.

## Project Overview

The study adopts a hybrid approach merging Network Science with Big Data Analytics to reconstruct a unified "Augmented Graph" where nodes possess both structural properties (PageRank, Betweenness) and operational attributes (Average Delay, Variance).

The analysis is conducted in four phases:

1. **Data Engineering:** Construction of the graph using PySpark. Static routes were enriched with operational data, identifying over 3,000 active routes missing from standard databases and validating the "Law of Large Numbers" in traffic-delay correlations.
2. **Robustness Analysis:** Statistical testing refutes the Fragility Paradox. High-centrality nodes exhibit superior stability, while operational volatility is concentrated in peripheral, low-degree nodes. Simulation of network disintegration confirms that removing high-delay nodes does not collapse the global component.
3. **Health-Based Clustering:** Application of K-Means and DBSCAN to segment airports into functional phenotypes. The analysis isolates a specific cluster of "Dysfunctional Periphery" nodes responsible for systemic inefficiency, distinct from the robust "Super-Hubs".
4. **Recommendation System:** Development of a "Smart Switch" engine using Locality Sensitive Hashing (LSH). The system identifies "Twin Airports"—structurally similar but operationally superior alternatives—demonstrating that strategic intervention on peripheral nodes yields significantly higher operational gains compared to optimizing core hubs.

## Key Findings

* **Centrality  Fragility:** Structural dominance correlates with operational stability; major hubs are resilient due to economies of scale.
* **Peripheral Instability:** Systemic risk is located in low-connectivity regional nodes rather than the global backbone.
* **Strategic Optimization:** The recommendation engine confirms that decongesting specific regional bottlenecks offers an average potential delay reduction of 13.8 minutes, significantly outperforming interventions on major hubs.

## Tech Stack

| Phase                          | Tool / Library                 |
| -------------------------------|--------------------------------|
| Large CSV Loading              | PySpark DataFrame             |
| Filtering / groupBy / Average  | PySpark DataFrame             |
| BTS–OpenFlights Join           | PySpark DataFrame             |
| Intermediate Storage           | Parquet / Pickle (Pandas)    |
| Network Analysis               | NetworkX (Python)            |
| Clustering / PCA               | scikit-learn                 |
| Plots / Graphs                 | matplotlib / seaborn         |
| Table Handling / Inspection    | pandas                       |


## Run The Code

To run the project locally, you need to download the datasets first. A helper script `setup_project_data.py` is provided. Run it with Python:

```bash
python setup_project_data.py
```
This will create the `data/` folder with the following structure:
```plaintext
data/
├─ anac_br/       # Brazil ANAC flight data CSVs
├─ bts_usa/       # USA BTS flight data CSVs
├─ caa_uk/        # UK CAA punctuality CSV
└─ openflights/   # OpenFlights airports and routes CSVs
```
After setting up the data, run:
```bash
python main.py
```

All plots will be generated in the plots/ folder.

**Optional: Colab Notebook**

If you prefer to use a notebook, open `main.ipynb` and click the Colab link at the top. This will launch the notebook in your browser. Press `Run All` and authorize Google Drive access to store outputs.