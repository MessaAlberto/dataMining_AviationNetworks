import os
import glob
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datasketch import MinHash, MinHashLSH
import random
import difflib

# --- CONFIGURAZIONE ---
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
CSV_DIR = os.path.join(BASE_DIR, "csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [DATA_DIR, PLOT_DIR, CSV_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)
os.chdir(DATA_DIR)

USE_CHECKPOINTS = False

# ==============================================================================
# FASE 1: DATA ENGINEERING & TRUTH CONSTRUCTION
# ==============================================================================
print("\n" + "="*60)
print("FASE 1: DATA ENGINEERING & INTEGRATION")
print("="*60)

ckpt_path_fase1 = os.path.join(CHECKPOINT_DIR, "fase1_data.pkl")

if USE_CHECKPOINTS and os.path.exists(ckpt_path_fase1):
    print(f"LOADING CHECKPOINT: {ckpt_path_fase1}")
    df_master_nodes, G = pd.read_pickle(ckpt_path_fase1)
else:
    # --- 1.1 CARICAMENTO STRUTTURALE (OpenFlights) ---
    print("[Structure] Loading OpenFlights Network...")
    df_airports = pd.read_csv("openflights/airports.csv", na_values="\\N")
    df_nodes = df_airports[
        df_airports["IATA"].str.match(r"^[A-Z]{3}$", na=False)
    ][["IATA", "Name", "City", "Country", "Latitude", "Longitude"]]

    df_routes = pd.read_csv("openflights/routes.csv", na_values="\\N")
    df_edges = df_routes[
        df_routes["SourceAirport"].str.match(r"^[A-Z]{3}$", na=False) &
        df_routes["DestAirport"].str.match(r"^[A-Z]{3}$", na=False) &
        (df_routes["Stops"] == 0)
    ][["SourceAirport", "DestAirport"]].rename(columns={"SourceAirport": "src", "DestAirport": "dst"})

    # --- 1.2 PULIZIA TOPOLOGICA ---
    print(f"   -> Raw Input: {len(df_nodes)} airports, {len(df_edges)} routes.")

    # A. Referential Integrity (Rimuove rotte verso aeroporti inesistenti)
    valid_iatas = set(df_nodes["IATA"].unique())
    mask_unknown = (~df_edges["src"].isin(valid_iatas)) | (~df_edges["dst"].isin(valid_iatas))
    num_unknown = mask_unknown.sum()

    if num_unknown > 0:
        print(f"   - Referential Integrity: Dropping {num_unknown} phantom routes (unknown endpoints).")
        df_edges = df_edges[~mask_unknown].copy()

    # B. Self-Loops
    mask_loops = df_edges["src"] == df_edges["dst"]
    num_loops = mask_loops.sum()

    if num_loops > 0:
        print(f"   - Self-Loops: Dropping {num_loops} circular routes.")
        df_edges = df_edges[~mask_loops].copy()

    # C. Duplicates (Rimuove voli multipli sulla stessa tratta operati da compagnie diverse)
    # Esempio: AA fa JFK-LHR, BA fa JFK-LHR -> diventano 1 sola rotta JFK-LHR
    before_dedup = len(df_edges)
    df_edges = df_edges.drop_duplicates(subset=["src", "dst"]).copy()
    num_dupes = before_dedup - len(df_edges)

    if num_dupes > 0:
        print(f"   - Deduplication: Merged {num_dupes} duplicate routes (multiple airlines on same edge).")

    # D. Isolated Nodes (Rimuove aeroporti senza voli per pulire il clustering)
    active_nodes = set(df_edges["src"]).union(set(df_edges["dst"]))
    initial_nodes_len = len(df_nodes)
    df_nodes = df_nodes[df_nodes["IATA"].isin(active_nodes)].copy()

    num_isolated = initial_nodes_len - len(df_nodes)

    if num_isolated > 0:
        print(f"   - Connectivity: Dropping {num_isolated} isolated airports (noise).")

    print(f"   -> Graph Cleaned: {len(df_nodes)} nodes, {len(df_edges)} edges.")

    print("[Structure] Building Graph G...")
    G = nx.from_pandas_edgelist(df_edges, "src", "dst", create_using=nx.DiGraph())
    G.add_nodes_from(df_nodes["IATA"])

    print(f"   -> Final Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.\n")

    # --- 1.3 INTEGRAZIONE DATI OPERATIVI (Delay) ---
    print("\n[Operations] Loading & Aligning Performance Data...")

    # A. USA (BTS)
    bts_files = glob.glob("bts_usa/*.csv")
    df_bts_metrics = pd.DataFrame()
    if bts_files:
        print("   -> Processing BTS (USA)...")
        df_bts = pd.concat((pd.read_csv(f, low_memory=False) for f in bts_files), ignore_index=True)
        df_bts = df_bts[(df_bts["Cancelled"] == 0) & (df_bts["Diverted"] == 0)]
        df_bts_metrics = df_bts.groupby("Origin")["DepDelayMinutes"].agg(
            avg_delay="mean", delay_variance="std", num_flights="count"
        ).reset_index().rename(columns={"Origin": "IATA"})

        # Filtro Validità
        valid_usa = df_bts_metrics[df_bts_metrics["IATA"].isin(active_nodes)]
        print(
            f"      - Raw Airports: {len(df_bts_metrics)} -> Matched in Graph: {len(valid_usa)} ({len(valid_usa)/len(df_bts_metrics):.1%})")
        df_bts_metrics = valid_usa

    # B. BRAZIL (ANAC)
    anac_files = glob.glob("anac_br/*.csv")
    df_anac_metrics = pd.DataFrame()
    if anac_files:
        print("   -> Processing ANAC (Brazil)...")
        icao_to_iata = df_airports.dropna(subset=["ICAO", "IATA"]).set_index("ICAO")["IATA"].to_dict()
        dfs = []
        for f in anac_files:
            try:
                df = pd.read_csv(f, sep=";", quotechar='"', encoding='utf-8', skiprows=1, low_memory=False)
            except:
                df = pd.read_csv(f, sep=";", quotechar='"', encoding='latin1', skiprows=1, low_memory=False)
            dfs.append(df)

        df_anac = pd.concat(dfs, ignore_index=True)
        # Calcolo ritardi
        t_real = pd.to_datetime(df_anac["Partida Real"], errors='coerce')
        t_sched = pd.to_datetime(df_anac["Partida Prevista"], errors='coerce')
        df_anac["delay"] = (t_real - t_sched).dt.total_seconds() / 60
        df_anac = df_anac[(df_anac["delay"].notna()) & (df_anac["Situação Voo"] == "REALIZADO")]
        df_anac["delay"] = df_anac["delay"].clip(lower=0)

        df_anac["IATA"] = df_anac["ICAO Aeródromo Origem"].map(icao_to_iata)
        df_anac_metrics = df_anac.groupby("IATA")["delay"].agg(
            avg_delay="mean", delay_variance="std", num_flights="count"
        ).reset_index()

        valid_br = df_anac_metrics[df_anac_metrics["IATA"].isin(active_nodes)]
        print(
            f"      - Raw Airports: {len(df_anac_metrics)} -> Matched in Graph: {len(valid_br)} ({len(valid_br)/len(df_anac_metrics):.1%})")
        df_anac_metrics = valid_br

    # C. UK (CAA)
    uk_files = glob.glob("caa_uk/*.csv")
    df_uk_metrics = pd.DataFrame()
    if uk_files:
        print("   -> Processing CAA (UK)...")
        # Dizionario IATA Robust
        manual_fix = {
            "GATWICK": "LGW",
            "HEATHROW": "LHR",
            "LUTON": "LTN",
            "STANSTED": "STN",
            "MANCHESTER": "MAN",
            "BIRMINGHAM": "BHX",
            "GLASGOW": "GLA",
            "EDINBURGH": "EDI",
            "BELFAST CITY (GEORGE BEST)": "BHD",
            "BELFAST INTERNATIONAL": "BFS",
            "EAST MIDLANDS INTERNATIONAL": "EMA",
            "NEWCASTLE": "NCL",
            "BRISTOL": "BRS",
            "LIVERPOOL (JOHN LENNON)": "LPL",
            "LEEDS BRADFORD": "LBA",
            "LONDON CITY": "LCY",
            "ABERDEEN": "ABZ",
            "SOUTHAMPTON": "SOU",
            "CARDIFF WALES": "CWL",
            "SOUTHEND": "SEN",
            "EXETER": "EXT",
            "ISLE OF MAN": "IOM",
            "JERSEY": "JER",
            "BOURNEMOUTH": "BOH",
            "TEESSIDE INTERNATIONAL AIRPORT": "MME"
        }
        clean_names = df_airports.assign(
            n=df_airports["Name"].str.upper().str.replace(" AIRPORT", "").str.strip()
        ).set_index("n")["IATA"].to_dict()

        def resolve_iata(name):
            n = str(name).strip().upper()
            if n in manual_fix:
                return manual_fix[n]
            n_simple = n.split("(")[0].strip()
            if n_simple in clean_names:
                return clean_names[n_simple]
            matches = difflib.get_close_matches(n_simple, clean_names.keys(), n=1, cutoff=0.7)
            return clean_names[matches[0]] if matches else None

        dfs_uk = []
        for f in uk_files:
            temp = pd.read_csv(f, encoding='latin1')
            temp = temp[temp["arrival_departure"] == "D"].copy()
            temp["IATA"] = temp["reporting_airport"].map(resolve_iata)

            # Weighted Avg Prep
            temp["avg"] = pd.to_numeric(temp["average_delay_mins"], errors='coerce').fillna(0)
            temp["cnt"] = pd.to_numeric(temp["number_flights_matched"], errors='coerce').fillna(0)
            temp["tot"] = temp["avg"] * temp["cnt"]
            dfs_uk.append(temp.dropna(subset=["IATA"]))

        if dfs_uk:
            df_uk_all = pd.concat(dfs_uk)
            df_uk_metrics = df_uk_all.groupby("IATA").agg(
                tot=("tot", "sum"), num_flights=("cnt", "sum"), delay_variance=("avg", "std")
            ).reset_index()
            df_uk_metrics["avg_delay"] = df_uk_metrics["tot"] / df_uk_metrics["num_flights"]
            df_uk_metrics = df_uk_metrics.drop(columns=["tot"])

            valid_uk = df_uk_metrics[df_uk_metrics["IATA"].isin(active_nodes)]
            print(
                f"      - Raw Airports: {len(df_uk_metrics)} -> Matched in Graph: {len(valid_uk)} ({len(valid_uk)/len(df_uk_metrics):.1%})")
            df_uk_metrics = valid_uk

    # --- 1.4 MERGE FINALE ---
    print("\n[Integration] Merging Operational Data into Structural Graph...")
    perf_dfs = [d for d in [df_bts_metrics, df_anac_metrics, df_uk_metrics] if not d.empty]

    if perf_dfs:
        df_perf_global = pd.concat(perf_dfs, ignore_index=True)
        # Aggregazione finale in caso di sovrapposizioni (es. media pesata semplice)
        df_perf_global = df_perf_global.groupby("IATA").mean().reset_index()

        # Merge LEFT: Manteniamo tutti i nodi del grafo, arricchiamo chi ha dati
        df_master_nodes = df_nodes.merge(df_perf_global, on="IATA", how="left")

        # Audit Finale Nodi
        total_nodes = len(df_master_nodes)
        with_data = df_master_nodes["avg_delay"].notna().sum()
        print(f"   -> GLOBAL COVERAGE: {with_data}/{total_nodes} nodes have delay data ({with_data/total_nodes:.1%})")

        # ---------------------------------------------------------
        # NUOVA SEZIONE: Audit Finale Archi (Edge Coverage)
        # ---------------------------------------------------------
        print("\n   [Diagnostic] Edge Coverage Check (Routes in Data vs Graph):")
        edges_in_graph = set(G.edges())  # Set di tuple (src, dst)
        real_routes = set()

        # 1. Recupero rotte reali USA (BTS)
        if 'df_bts' in locals() and not df_bts.empty:
            # BTS ha colonne Origin e Dest già in formato IATA
            bts_routes = set(zip(df_bts["Origin"], df_bts["Dest"]))
            real_routes.update(bts_routes)

        # 2. Recupero rotte reali Brasile (ANAC)
        if 'df_anac' in locals() and not df_anac.empty:
            # ANAC usa ICAO per la destinazione, dobbiamo convertirla in IATA
            # Recuperiamo il dizionario se esiste, altrimenti lo ricreiamo
            if 'icao_to_iata' not in locals():
                icao_to_iata = df_airports.dropna(subset=["ICAO", "IATA"]).set_index("ICAO")["IATA"].to_dict()

            # Creiamo colonna temporanea per il check
            df_anac["Dest_IATA"] = df_anac["ICAO Aeródromo Destino"].map(icao_to_iata)
            anac_valid = df_anac.dropna(subset=["IATA", "Dest_IATA"])
            anac_routes = set(zip(anac_valid["IATA"], anac_valid["Dest_IATA"]))
            real_routes.update(anac_routes)

        # 3. Calcolo differenze
        if real_routes:
            total_real = len(real_routes)
            covered_edges = len([r for r in real_routes if r in edges_in_graph])
            missing_edges_set = real_routes - edges_in_graph
            missing_count = len(missing_edges_set)

            print(f"      - Unique Real Routes Observed: {total_real}")
            print(f"      - Matched in OpenFlights Graph: {covered_edges} ({covered_edges/total_real:.1%})")
            print(f"      - MISSING EDGES: {missing_count} ({missing_count/total_real:.1%})")

            if missing_count > 0:
                print(f"        (Note: These are likely new routes created after OpenFlights DB)")
                print("\n      [Analysis & Enrichment] Processing missing edges...")

                graph_nodes = set(G.nodes())
                missing_data = []
                counts = {"Both_Nodes_Exist": 0, "Only_Src_Exists": 0, "Only_Dst_Exists": 0, "Neither_Exists": 0}

                added_count = 0
                for src, dst in missing_edges_set:
                    src_exists = src in graph_nodes
                    dst_exists = dst in graph_nodes

                    status = "Neither_Exists"
                    if src_exists and dst_exists:
                        status = "Both_Nodes_Exist"
                        G.add_edge(src, dst)  # Aggiungiamo l'arco al Grafo NetworkX!
                        added_count += 1
                    elif src_exists:
                        status = "Only_Src_Exists"
                    elif dst_exists:
                        status = "Only_Dst_Exists"

                    counts[status] += 1
                    missing_data.append({"src": src, "dst": dst, "reason": status})

                # Stampa Statistiche
                print(
                    f"      - New Routes (Both Airports in Graph): {counts['Both_Nodes_Exist']} ({counts['Both_Nodes_Exist']/missing_count:.1%})")
                print(
                    f"      ---> SUCCESS: Added {added_count} new routes to the Graph! The network is now up-to-date.")
                print(f"      ---> New Graph Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
                print(
                    f"      - Dead End (Only Source in Graph):     {counts['Only_Src_Exists']} ({counts['Only_Src_Exists']/missing_count:.1%})")
                print(
                    f"      - Incoming (Only Dest in Graph):       {counts['Only_Dst_Exists']} ({counts['Only_Dst_Exists']/missing_count:.1%})")
                print(
                    f"      - Unknown (Neither in Graph):          {counts['Neither_Exists']} ({counts['Neither_Exists']/missing_count:.1%})")

                # Esporta CSV
                miss_csv_path = os.path.join(CSV_DIR, "missing_routes_analysis.csv")
                pd.DataFrame(missing_data).to_csv(miss_csv_path, index=False)
                print(f"      -> Detailed missing routes saved to: {miss_csv_path}")

        else:
            print("      - No raw route data available in memory for edge verification.")
        # ---------------------------------------------------------

        print("\nSaving Checkpoint Phase 1...")
        pd.to_pickle((df_master_nodes, G), ckpt_path_fase1)
    else:
        raise ValueError("No performance data found! Check input folders.")

# Plot 1
plt.figure(figsize=(10, 6))
sns.histplot(df_master_nodes['avg_delay'].dropna(), bins=50, kde=True, color='skyblue')
plt.title('Global Average Delay Distribution')
plt.savefig(os.path.join(PLOT_DIR, "01_delay_distribution.png"))
plt.close()

# ==============================================================================
# FASE 2: NETWORK ANALYSIS (Metrics & Simulation)
# ==============================================================================
print("\n" + "="*60)
print("FASE 2: NETWORK METRICS & ROBUSTNESS")
print("="*60)

ckpt_path_fase2 = os.path.join(CHECKPOINT_DIR, "fase2_metrics.pkl")

if USE_CHECKPOINTS and os.path.exists(ckpt_path_fase2):
    print(f"LOADING CHECKPOINT: {ckpt_path_fase2}")
    df_master_nodes = pd.read_pickle(ckpt_path_fase2)
else:
    print("Calculating Centrality Metrics...")
    # PageRank
    df_master_nodes["pagerank"] = df_master_nodes["IATA"].map(nx.pagerank(G))
    # Betweenness (Approximation k=500 per performance)
    df_master_nodes["betweenness"] = df_master_nodes["IATA"].map(nx.betweenness_centrality(G, k=500))
    # Degree
    df_master_nodes["degree"] = df_master_nodes["IATA"].map(dict(G.degree()))

    pd.to_pickle(df_master_nodes, ckpt_path_fase2)

# 2.1 Power Law Check
degrees = [d for n, d in G.degree() if d > 0]
degree_counts = pd.Series(degrees).value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.loglog(degree_counts.index, degree_counts.values, 'bo', alpha=0.6)
plt.title("Degree Distribution (Log-Log Scale)")
plt.xlabel("Degree (k)")
plt.ylabel("Count")
plt.savefig(os.path.join(PLOT_DIR, "02a_power_law.png"))
plt.close()

# 2.2 Correlation Analysis
df_valid = df_master_nodes.dropna(subset=["avg_delay", "pagerank"])
corr = df_valid["pagerank"].corr(df_valid["avg_delay"])
print(f"-> Correlation (Centrality vs Delay): {corr:.4f}")

plt.figure(figsize=(10, 6))
sns.regplot(x='pagerank', y='avg_delay', data=df_valid,
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title(f'Centrality vs Delay (Corr: {corr:.2f})')
plt.ylim(0, 100)
plt.savefig(os.path.join(PLOT_DIR, "02_correlation.png"))
plt.close()

# 2.3 Fragility Simulation (Crash Curve)
print("Running Fragility Simulation (Random vs Hubs vs Bottlenecks)...")


def get_gc_size(g):
    return len(max(nx.weakly_connected_components(g), key=len)) if len(g) > 0 else 0


nodes_list = list(G.nodes())
initial_size = get_gc_size(G)
fractions = np.linspace(0, 0.2, 10)

results = {"Random": [], "Structural (Hubs)": [], "Operational (Delays)": []}
sorted_pr = df_master_nodes.sort_values("pagerank", ascending=False)["IATA"].tolist()
sorted_delay = df_master_nodes.sort_values("avg_delay", ascending=False)["IATA"].tolist()

for f in fractions:
    n_rem = int(len(nodes_list) * f)

    # A. Random
    G_tmp = G.copy()
    G_tmp.remove_nodes_from(random.sample(nodes_list, n_rem))
    results["Random"].append(get_gc_size(G_tmp) / initial_size)

    # B. Hubs
    G_tmp = G.copy()
    targets = [n for n in sorted_pr if n in G_tmp][:n_rem]
    G_tmp.remove_nodes_from(targets)
    results["Structural (Hubs)"].append(get_gc_size(G_tmp) / initial_size)

    # C. Operational
    G_tmp = G.copy()
    targets = [n for n in sorted_delay if n in G_tmp][:n_rem]
    G_tmp.remove_nodes_from(targets)
    results["Operational (Delays)"].append(get_gc_size(G_tmp) / initial_size)

plt.figure(figsize=(10, 6))
plt.plot(fractions, results["Random"], '--o', label="Random Failure")
plt.plot(fractions, results["Structural (Hubs)"], '-s', color='red', label="Targeted Attack (Hubs)")
plt.plot(fractions, results["Operational (Delays)"], '-^', color='orange', label="Operational Collapse (Delay)")
plt.title("Network Robustness Simulation")
plt.xlabel("Fraction Removed")
plt.ylabel("Giant Component Size (Normalized)")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "02b_simulation.png"))
plt.close()

# ==============================================================================
# FASE 3: CLUSTERING & PROFILING
# ==============================================================================
print("\n" + "="*60)
print("FASE 3: UNSUPERVISED CLUSTERING (PCA + KMEANS)")
print("="*60)

features = ["pagerank", "betweenness", "avg_delay", "delay_variance", "degree"]
df_cluster = df_master_nodes.dropna(subset=features).copy()

# Scaling & PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_cluster["pca_1"], df_cluster["pca_2"] = X_pca[:, 0], X_pca[:, 1]

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster["cluster"] = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, df_cluster["cluster"])
print(f"-> Clustering Silhouette Score: {sil:.4f}")

# Merge results back
df_master_nodes = df_master_nodes.drop(columns=["cluster"], errors='ignore').merge(
    df_cluster[["IATA", "cluster", "pca_1", "pca_2"]], on="IATA", how="left"
)

# Plot Clusters PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', data=df_cluster, palette='viridis', s=80)
plt.title(f'Airport Clusters (PCA Space)')
plt.savefig(os.path.join(PLOT_DIR, "03_pca_clusters.png"))
plt.close()

# Plot Geographic
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='cluster',
                data=df_master_nodes.dropna(subset=["cluster"]), palette='viridis', s=20)
plt.title('Geographic Distribution of Clusters')
plt.savefig(os.path.join(PLOT_DIR, "04_geo_clusters.png"))
plt.close()

# ==============================================================================
# FASE 4: RECOMMENDATION SYSTEM (LSH)
# ==============================================================================
print("\n" + "="*60)
print("FASE 4: LSH RECOMMENDATIONS")
print("="*60)

# Identify Bottleneck Cluster
cluster_stats = df_cluster.groupby("cluster")["avg_delay"].mean()
bad_cluster = cluster_stats.idxmax()
print(f"Bottleneck Cluster: #{bad_cluster} (Avg Delay: {cluster_stats[bad_cluster]:.1f} min)")

# Setup LSH
lsh = MinHashLSH(threshold=0.5, num_perm=128)
minhashes = {}
print("Indexing Graph Connectivity...")
for n in G.nodes():
    dests = list(G.successors(n))
    if dests:
        m = MinHash(num_perm=128)
        for d in dests:
            m.update(str(d).encode('utf8'))
        minhashes[n] = m
        lsh.insert(n, m)

# Find Alternatives
bottlenecks = df_master_nodes[df_master_nodes["cluster"] == bad_cluster]["IATA"].tolist()
recs = []

print(f"Scanning {len(bottlenecks)} bottleneck airports for alternatives...")
for b in bottlenecks:
    if b in minhashes:
        candidates = lsh.query(minhashes[b])
        for c in candidates:
            if c != b:
                # Check metrics of candidate
                row = df_master_nodes[df_master_nodes["IATA"] == c]
                if not row.empty:
                    c_cluster = row.iloc[0]["cluster"]
                    c_delay = row.iloc[0]["avg_delay"]

                    # Recommendation Logic: Must be in a better cluster or significantly faster
                    if pd.notna(c_cluster) and c_cluster != bad_cluster:
                        recs.append({
                            "Bottleneck": b, "Alternative": c,
                            "Alt_Delay": c_delay
                        })

df_recs = pd.DataFrame(recs)
if not df_recs.empty:
    df_recs = df_recs.sort_values("Alt_Delay")
    print("\n--- TOP RECOMMENDATIONS ---")
    print(df_recs.head(10).to_string(index=False))
else:
    print("No recommendations found.")

# ==============================================================================
# FASE 5: EXPORT
# ==============================================================================
print("\n" + "="*60)
print("FASE 5: SAVING RESULTS")
print("="*60)

# Export Summary Stats
summary = df_master_nodes.groupby("cluster")[features].mean()
print("Cluster Profiles:\n", summary)

# Save CSVs
df_master_nodes.to_csv(os.path.join(CSV_DIR, "final_results_global.csv"), index=False)
df_recs.to_csv(os.path.join(CSV_DIR, "recommendations.csv"), index=False)
df_cluster.to_csv(os.path.join(CSV_DIR, "clustering_debug.csv"), index=False)

print(f"Analysis Complete. Results saved in: {CSV_DIR}")
