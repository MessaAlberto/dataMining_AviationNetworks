import difflib
import glob
import os
import random
import time
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as shc
from scipy import stats
from datasketch import MinHash, MinHashLSH

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Warnings and plotting config
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
CSV_DIR = os.path.join(BASE_DIR, "csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [DATA_DIR, PLOT_DIR, CSV_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

os.chdir(DATA_DIR)

# Config
USE_CHECKPOINTS = True

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
    # --- 1.1 STRUCTURE BUILDING (OpenFlights) ---
    print("Loading OpenFlights Network...")
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

    # --- 1.2 TOPOLOGICAL CLEANING ---
    print(f"Raw Input: {len(df_nodes)} airports, {len(df_edges)} routes.")

    # A. Referential Integrity (Removes routes to non-existent airports)
    valid_iatas = set(df_nodes["IATA"].unique())
    mask_unknown = (~df_edges["src"].isin(valid_iatas)) | (~df_edges["dst"].isin(valid_iatas))
    num_unknown = mask_unknown.sum()

    if num_unknown > 0:
        print(f"    - Referential Integrity: Dropping {num_unknown} phantom routes (unknown airports).")
        df_edges = df_edges[~mask_unknown].copy()

    # B. Self-Loops
    mask_loops = df_edges["src"] == df_edges["dst"]
    num_loops = mask_loops.sum()

    if num_loops > 0:
        print(f"    - Self-Loops: Dropping {num_loops} circular routes (src == dst).")
        df_edges = df_edges[~mask_loops].copy()

    # C. Duplicates (Removes multiple flights on the same route operated by different airlines)
    # Example: AA flies JFK-LHR, BA flies JFK-LHR -> becomes 1 single route JFK-LHR
    before_dedup = len(df_edges)
    df_edges = df_edges.drop_duplicates(subset=["src", "dst"]).copy()
    num_dupes = before_dedup - len(df_edges)

    if num_dupes > 0:
        print(f"    - Deduplication: Merged {num_dupes} duplicate routes (multiple airlines on same edge).")

    # D. Isolated Nodes (Removes airports without flights to clean clustering)
    active_nodes = set(df_edges["src"]).union(set(df_edges["dst"]))
    initial_nodes_len = len(df_nodes)
    df_nodes = df_nodes[df_nodes["IATA"].isin(active_nodes)].copy()

    num_isolated = initial_nodes_len - len(df_nodes)

    if num_isolated > 0:
        print(f"    - Connectivity: Dropping {num_isolated} isolated airports (noise).")

    print(f"Graph Cleaned: {len(df_nodes)} nodes, {len(df_edges)} edges.")

    print("\nBuilding Graph G...")
    G = nx.from_pandas_edgelist(df_edges, "src", "dst", create_using=nx.DiGraph())
    G.add_nodes_from(df_nodes["IATA"])

    print(f"    -> Final Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.\n")

    # --- 1.3 OPERATIONS DATA INTEGRATION (Delay) ---
    print("\nLoading & Aligning Performance Data...")

    # A. USA (BTS)
    bts_files = glob.glob("bts_usa/*.csv")
    df_bts_metrics = pd.DataFrame()
    if bts_files:
        print("Processing BTS (USA) Data...")
        df_bts = pd.concat((pd.read_csv(f, low_memory=False) for f in bts_files), ignore_index=True)
        print(f"    Finished loading BTS data: {len(df_bts)} records.")
        df_bts = df_bts[(df_bts["Cancelled"] == 0) & (df_bts["Diverted"] == 0)]
        print(f"    Filtered BTS data: {len(df_bts)} records after removing cancelled/diverted flights.")
        df_bts_metrics = df_bts.groupby("Origin")["DepDelayMinutes"].agg(
            avg_delay="mean", delay_variance="std", num_flights="count"
        ).reset_index().rename(columns={"Origin": "IATA"})
        print(f"    Aggregated BTS metrics: {len(df_bts_metrics)} unique origin airports.")

        # Validity Filter
        valid_usa = df_bts_metrics[df_bts_metrics["IATA"].isin(active_nodes)]
        print(
            f"-> Raw Airports: {len(df_bts_metrics)} -> Matched in Graph: {len(valid_usa)} ({len(valid_usa)/len(df_bts_metrics):.1%})")
        df_bts_metrics = valid_usa

    # B. BRAZIL (ANAC)
    anac_files = glob.glob("anac_br/*.csv")
    df_anac_metrics = pd.DataFrame()
    if anac_files:
        print("\nProcessing ANAC (Brazil) Data...")
        icao_to_iata = df_airports.dropna(subset=["ICAO", "IATA"]).set_index("ICAO")["IATA"].to_dict()
        dfs = []
        for f in anac_files:
            try:
                df = pd.read_csv(f, sep=";", quotechar='"', encoding='utf-8', skiprows=1, low_memory=False)
            except:
                df = pd.read_csv(f, sep=";", quotechar='"', encoding='latin1', skiprows=1, low_memory=False)
            dfs.append(df)

        df_anac = pd.concat(dfs, ignore_index=True)
        print(f"    Finished loading ANAC data: {len(df_anac)} records.")
        t_real = pd.to_datetime(df_anac["Partida Real"], errors='coerce')
        t_sched = pd.to_datetime(df_anac["Partida Prevista"], errors='coerce')
        df_anac["delay"] = (t_real - t_sched).dt.total_seconds() / 60
        df_anac = df_anac[(df_anac["delay"].notna()) & (df_anac["Situação Voo"] == "REALIZADO")]
        df_anac["delay"] = df_anac["delay"].clip(lower=0)
        print(f"    Filtered ANAC data: {len(df_anac)} records after removing invalid delays and non-realized flights.")

        df_anac["IATA"] = df_anac["ICAO Aeródromo Origem"].map(icao_to_iata)
        df_anac_metrics = df_anac.groupby("IATA")["delay"].agg(
            avg_delay="mean", delay_variance="std", num_flights="count"
        ).reset_index()
        print(f"    Aggregated ANAC metrics: {len(df_anac_metrics)} unique origin airports.")

        valid_br = df_anac_metrics[df_anac_metrics["IATA"].isin(active_nodes)]
        print(
            f"-> Raw Airports: {len(df_anac_metrics)} -> Matched in Graph: {len(valid_br)} ({len(valid_br)/len(df_anac_metrics):.1%})")
        df_anac_metrics = valid_br

    # C. UK (CAA)
    uk_files = glob.glob("caa_uk/*.csv")
    df_uk_metrics = pd.DataFrame()
    if uk_files:
        print("\nProcessing CAA (UK) Data...")
        # Robust IATA Dictionary
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
            print(f"    Aggregated CAA (UK) metrics: {len(df_uk_metrics)} unique origin airports.")
            df_uk_metrics = df_uk_metrics.drop(columns=["tot"])

            valid_uk = df_uk_metrics[df_uk_metrics["IATA"].isin(active_nodes)]
            print(
                f"-> Raw Airports: {len(df_uk_metrics)} -> Matched in Graph: {len(valid_uk)} ({len(valid_uk)/len(df_uk_metrics):.1%})")
            df_uk_metrics = valid_uk

    # --- 1.4 FINAL MERGE ---
    print("\nMerging Operational Data into Structural Graph...")
    perf_dfs = [d for d in [df_bts_metrics, df_anac_metrics, df_uk_metrics] if not d.empty]

    if perf_dfs:
        df_perf_global = pd.concat(perf_dfs, ignore_index=True)
        # Final aggregation in case of overlaps (e.g., simple weighted average)
        df_perf_global = df_perf_global.groupby("IATA").mean().reset_index()

        # Merge LEFT: Keep all graph nodes, enrich those with data
        df_master_nodes = df_nodes.merge(df_perf_global, on="IATA", how="left")

        # Final Node Audit
        total_nodes = len(df_master_nodes)
        with_data = df_master_nodes["avg_delay"].notna().sum()
        print(f"    - Airport coverage: {with_data}/{total_nodes} nodes have delay data ({with_data/total_nodes:.1%})")

        # Edge validation
        edges_in_graph = set(G.edges())
        real_routes = set()

        # 1. Real routes USA (BTS)
        if 'df_bts' in locals() and not df_bts.empty:
            # BTS has Origin and Dest columns already in IATA format
            bts_routes = set(zip(df_bts["Origin"], df_bts["Dest"]))
            real_routes.update(bts_routes)

        # 2. Real routes Brazil (ANAC)
        if 'df_anac' in locals() and not df_anac.empty:
            # ANAC uses ICAO for the destination, we need to convert it to IATA
            if 'icao_to_iata' not in locals():
                icao_to_iata = df_airports.dropna(subset=["ICAO", "IATA"]).set_index("ICAO")["IATA"].to_dict()

            # Tmp columns
            df_anac["Dest_IATA"] = df_anac["ICAO Aeródromo Destino"].map(icao_to_iata)
            anac_valid = df_anac.dropna(subset=["IATA", "Dest_IATA"])
            anac_routes = set(zip(anac_valid["IATA"], anac_valid["Dest_IATA"]))
            real_routes.update(anac_routes)

        # 3. Calculate differences
        if real_routes:
            total_real = len(real_routes)
            covered_edges = len([r for r in real_routes if r in edges_in_graph])
            missing_edges_set = real_routes - edges_in_graph
            missing_count = len(missing_edges_set)

            print(f"    - Unique Real Routes Observed: {total_real}")
            print(f"    - Matched in OpenFlights Graph: {covered_edges} ({covered_edges/total_real:.1%})")
            print(f"    - MISSING EDGES: {missing_count} ({missing_count/total_real:.1%})")

            if missing_count > 0:
                print(f"        (Note: These are likely new routes created after OpenFlights DB)")
                print("\nProcessing missing edges...")

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
                        G.add_edge(src, dst)
                        added_count += 1
                    elif src_exists:
                        status = "Only_Src_Exists"
                    elif dst_exists:
                        status = "Only_Dst_Exists"

                    counts[status] += 1
                    missing_data.append({"src": src, "dst": dst, "reason": status})

                # Print Statistics
                print(
                    f"    - New Routes (src/dst both in Graph): {counts['Both_Nodes_Exist']} ({counts['Both_Nodes_Exist']/missing_count:.1%})")
                print(
                    f"     -> SUCCESS: Added {added_count} new routes to the Graph! The network is now up-to-date.")
                print(f"     -> New Graph Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
                print(
                    f"    - Dead End (Only Source in Graph):     {counts['Only_Src_Exists']} ({counts['Only_Src_Exists']/missing_count:.1%})")
                print(
                    f"    - Incoming (Only Dest in Graph):       {counts['Only_Dst_Exists']} ({counts['Only_Dst_Exists']/missing_count:.1%})")
                print(
                    f"    - Unknown (Neither in Graph):          {counts['Neither_Exists']} ({counts['Neither_Exists']/missing_count:.1%})")

                # Export CSV
                miss_csv_path = os.path.join(CSV_DIR, "missing_routes_analysis.csv")
                pd.DataFrame(missing_data).to_csv(miss_csv_path, index=False)
                print(f"-> Detailed missing routes saved to: {miss_csv_path}")

        else:
            print("\nNo raw route data available in memory for edge verification.")
        # ---------------------------------------------------------

        print("\nSaving Checkpoint Phase 1...")
        pd.to_pickle((df_master_nodes, G), ckpt_path_fase1)
    else:
        raise ValueError("No performance data found! Check input folders.")

print("Generating Phase 1 Diagnostic Plots...")
# --- PLOT 1: DELAY DISTRIBUTION ---
plt.figure(figsize=(10, 6))
sns.histplot(df_master_nodes['avg_delay'].dropna(), bins=50, kde=True)
plt.title('Global Average Delay Distribution')
plt.savefig(os.path.join(PLOT_DIR, "01a_delay_distribution.png"))
plt.close()
print("-> Saved: 01a_delay_distribution.png")

# --- PLOT 2: GEOGRAPHIC COVERAGE (The "World Map") ---
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x='Longitude', y='Latitude',
    data=df_master_nodes,
    color='lightgray', s=10, alpha=0.5, label='Structural Backbone'
)

df_with_data = df_master_nodes.dropna(subset=['avg_delay'])
sns.scatterplot(
    x='Longitude', y='Latitude',
    data=df_with_data,
    hue='avg_delay', palette='viridis', s=20, alpha=1.0
)

plt.title(f'Network Coverage: {len(df_with_data)} Operational Nodes vs {len(df_master_nodes)} Structural Nodes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='lower left', fontsize='small')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig(os.path.join(PLOT_DIR, "01b_geo_coverage.png"))
plt.close()
print("-> Saved: 01b_geo_coverage.png")


# --- PLOT 3: VOLUME VS DELAY (Efficiency Check) ---
if not df_with_data.empty and 'num_flights' in df_with_data.columns:
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x='num_flights', y='avg_delay',
        data=df_with_data,
        alpha=0.6, edgecolor='w', s=60
    )

    plt.xscale('log')
    plt.title('Traffic Volume vs Average Delay')
    plt.xlabel('Number of Recorded Flights (Log Scale)')
    plt.ylabel('Average Delay (min)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(PLOT_DIR, "01c_volume_vs_delay.png"))
    plt.close()
    print("-> Saved: 01c_volume_vs_delay.png")


# --- PLOT 4: RELIABILITY (Mean Delay vs Variance) ---
if not df_with_data.empty and 'delay_variance' in df_with_data.columns:
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x='avg_delay', y='delay_variance',
        data=df_with_data,
        alpha=0.6, color='coral', edgecolor='w', s=60
    )

    sns.regplot(
        x='avg_delay', y='delay_variance',
        data=df_with_data, scatter=False, color='red', line_kws={'linestyle': '--'}
    )

    plt.title('Delay Reliability: Mean vs Variance')
    plt.xlabel('Average Delay (min)')
    plt.ylabel('Delay Standard Deviation (min)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "01d_reliability_check.png"))
    plt.close()
    print("-> Saved: 01d_reliability_check.png")

# ==============================================================================
# FASE 2: NETWORK ANALYSIS & THE "FRAGILITY PARADOX"
# ==============================================================================
print("\n" + "="*60)
print("FASE 2: TOPOLOGICAL METRICS & ROBUSTNESS SIMULATION")
print("="*60)

ckpt_path_fase2 = os.path.join(CHECKPOINT_DIR, "fase2_metrics.pkl")

# --- 2.1 TOPOLOGICAL METRICS CALCULATION ---
if USE_CHECKPOINTS and os.path.exists(ckpt_path_fase2):
    print(f"LOADING CHECKPOINT: {ckpt_path_fase2}")
    df_master_nodes = pd.read_pickle(ckpt_path_fase2)

    # Recalculate global metrics not stored in DF
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"   -> Global Assortativity Coefficient: {assortativity:.4f}")

else:
    print("Calculating Network Centrality Measures...")

    # A. GIANT CONNECTED COMPONENT (GCC) EXTRACTION
    # Path-based metrics (Closeness, Diameter) require a connected graph.
    print("    - Extracting Giant Connected Component (GCC) for path metrics...")
    gcc_nodes = max(nx.connected_components(G.to_undirected()), key=len)
    G_gcc = G.subgraph(gcc_nodes).copy()

    # 1. Degree Centrality (Connectivity)
    print("    - Calculating In-Degree and Out-Degree...")
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())
    total_degree_dict = dict(G.degree())  # Sum of In + Out

    df_master_nodes["in_degree"] = df_master_nodes["IATA"].map(in_degree_dict)
    df_master_nodes["out_degree"] = df_master_nodes["IATA"].map(out_degree_dict)
    df_master_nodes["degree"] = df_master_nodes["IATA"].map(total_degree_dict)

    # 2. PageRank (Global Importance)
    print("    - Calculating PageRank (alpha=0.85)...")
    pagerank_dict = nx.pagerank(G, alpha=0.85)
    df_master_nodes["pagerank"] = df_master_nodes["IATA"].map(pagerank_dict)

    # 3. Betweenness Centrality (Bottleneck Potential)
    print("    - Calculating Betweenness (Approx k=500)...")
    betweenness_dict = nx.betweenness_centrality(G, k=500, normalized=True, seed=42)
    df_master_nodes["betweenness"] = df_master_nodes["IATA"].map(betweenness_dict)

    # 4. Closeness Centrality (Accessibility Speed) - NEW!
    # Calculated on GCC to avoid infinite distance errors, then mapped back.
    print("    - Calculating Closeness Centrality (on GCC)...")
    closeness_gcc = nx.closeness_centrality(G_gcc)
    df_master_nodes["closeness"] = df_master_nodes["IATA"].map(closeness_gcc)

    # 5. Assortativity & Mixing Patterns
    # r < 0: Disassortative (Hubs connect to spokes) -> Typical of Hub & Spoke
    print("    - Calculating Global Assortativity Coefficient...")
    assortativity = nx.degree_assortativity_coefficient(G)

    # 6. Average Nearest Neighbor Degree (ANND) - NEW!
    # We calculate k_nn to verify if high-degree nodes connect to low-degree nodes.
    print("    - Calculating Average Nearest Neighbor Degree (ANND)...")
    knn_dict = nx.average_neighbor_degree(G)
    df_master_nodes["knn"] = df_master_nodes["IATA"].map(knn_dict)

    # ---------------------------
    # PRINT SUMMARY STATISTICS
    # ---------------------------
    print("\nNetwork Centrality Summary:")

    # GCC info
    print(f"    - GCC Size: {len(G_gcc)} nodes ({len(G_gcc)/len(G):.1%} of total)")

    # Degree stats
    print(f"    - In-Degree: min={df_master_nodes['in_degree'].min()}, max={df_master_nodes['in_degree'].max()}, mean={df_master_nodes['in_degree'].mean():.2f}")
    print(f"    - Out-Degree: min={df_master_nodes['out_degree'].min()}, max={df_master_nodes['out_degree'].max()}, mean={df_master_nodes['out_degree'].mean():.2f}")
    print(f"    - Total Degree: min={df_master_nodes['degree'].min()}, max={df_master_nodes['degree'].max()}, mean={df_master_nodes['degree'].mean():.2f}")

    # Pagerank stats
    print(f"    - PageRank: min={df_master_nodes['pagerank'].min():.6f}, max={df_master_nodes['pagerank'].max():.6f}, mean={df_master_nodes['pagerank'].mean():.6f}")

    # Betweenness stats
    print(f"    - Betweenness: min={df_master_nodes['betweenness'].min():.6f}, max={df_master_nodes['betweenness'].max():.6f}, mean={df_master_nodes['betweenness'].mean():.6f}")

    # Closeness stats
    if 'closeness' in df_master_nodes.columns:
        print(f"    - Closeness: min={df_master_nodes['closeness'].min():.4f}, max={df_master_nodes['closeness'].max():.4f}, mean={df_master_nodes['closeness'].mean():.4f}")

    # Assortativity
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"    - Global Assortativity Coefficient: {assortativity:.4f}")

    # Average nearest neighbor degree
    knn_stats = df_master_nodes['knn'].describe()
    print(f"    - Average Neighbor Degree: min={knn_stats['min']:.2f}, max={knn_stats['max']:.2f}, mean={knn_stats['mean']:.2f}")

    # Optional: top-k nodes by key metrics
    top_k = 5
    print("\nTop nodes by PageRank:")
    print(df_master_nodes.nlargest(top_k, "pagerank")[["IATA", "pagerank"]])

    print("\nTop nodes by Betweenness:")
    print(df_master_nodes.nlargest(top_k, "betweenness")[["IATA", "betweenness"]])

    print("\nTop nodes by Closeness:")
    if 'closeness' in df_master_nodes.columns:
        print(df_master_nodes.nlargest(top_k, "closeness")[["IATA", "closeness"]])

    print("\nSaving Checkpoint Phase 2...")
    pd.to_pickle(df_master_nodes, ckpt_path_fase2)

# --- 2.2 ASSORTATIVITY ANALYSIS (ANND PLOT) ---
# This visualizes the correlation between a node's degree and its neighbors' degree.
print("\nDeep Dive on Assortativity (ANND)...")

# Plot: Out-Degree (Hub capacity) vs Average In-Degree of neighbors (Destination popularity)
df_annd = df_master_nodes.dropna(subset=["out_degree", "knn"])
# Filter noise: remove nodes with degree 0 or very small
df_annd = df_annd[df_annd["out_degree"] > 0]

k_vs_knn = df_annd.groupby("out_degree")["knn"].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='out_degree', y='knn', data=k_vs_knn, color='purple', alpha=0.6, label='Observed Data')

# Trend line (linear regression in log-log space)
if len(k_vs_knn) > 1:
    z = np.polyfit(k_vs_knn['out_degree'], k_vs_knn['knn'], 1)
    p = np.poly1d(z)
    plt.plot(k_vs_knn['out_degree'], p(k_vs_knn['out_degree']), "r--", label=f'Trend Slope: {z[0]:.4f}')

plt.title("Assortativity: Out-Degree vs Neighbor In-Degree")
plt.xlabel("Node Out-Degree ($k_{out}$)")
plt.ylabel("Average Neighbor In-Degree ($k_{nn}^{in}$)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(PLOT_DIR, "02a_assortativity_annd.png"))
plt.close()
print("-> Saved: 02a_assortativity_annd.png")

# --- 2.3 SCALE-FREE VALIDATION (DUAL METHOD) ---
# We compare two methods to estimate the Scaling Exponent (Gamma):
# A. Classic PDF: Plotting P(k) vs k. Intuitive but noisy in the tail.
# B. Robust CCDF: Plotting P(K >= k) vs k. Eliminates noise and binning bias.

print("\nVerifying Scale-Free Property (PDF vs CCDF)...")
degrees = [d for n, d in G.degree() if d > 0]

# --- METHOD A: CLASSIC LOG-LOG PDF + REGRESSION ---
print("    - Method A: Classic Degree Distribution (PDF)...")
degree_counts = pd.Series(degrees).value_counts().sort_index()
x_pdf = degree_counts.index.values
y_pdf = degree_counts.values / sum(degree_counts.values)  # P(k) probability

# Linear Regression on Log-Log data
# log(P(k)) = -gamma * log(k) + c
log_x_pdf = np.log(x_pdf)
log_y_pdf = np.log(y_pdf)
coeffs_pdf = np.polyfit(log_x_pdf, log_y_pdf, 1)
gamma_pdf = -coeffs_pdf[0]  # Slope is -gamma

print(f"      Estimated Gamma (PDF Method): {gamma_pdf:.2f}")

plt.figure(figsize=(8, 6))
plt.loglog(x_pdf, y_pdf, 'bo', alpha=0.5, label='Observed P(k)')
plt.loglog(x_pdf, np.exp(np.polyval(coeffs_pdf, log_x_pdf)), 'r--', linewidth=2,
           label=f'Fit ($\gamma={gamma_pdf:.2f}$)')
plt.title(f"A. Classic Degree Distribution (PDF)\nGamma ~ {gamma_pdf:.2f}")
plt.xlabel("Degree (k)")
plt.ylabel("Probability P(k)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.savefig(os.path.join(PLOT_DIR, "02b_power_law_pdf_classic.png"))
plt.close()


# --- METHOD B: ROBUST CCDF (Complementary Cumulative Distribution) ---
print("    - Method B: Robust CCDF (Cumulative)...")

def get_ccdf_distribution(degrees):
    """
    Calculates P(K >= k).
    Method: Sort data and calculate rank frequencies.
    """
    degrees = np.array(degrees)
    degrees_sorted = np.sort(degrees)

    n = len(degrees_sorted)
    ranks = np.arange(n)
    # fraction of nodes with degree >= k
    ccdf = 1 - (ranks / n)

    return degrees_sorted, ccdf


x_ccdf, y_ccdf = get_ccdf_distribution(degrees)

# Linear Regression on CCDF
# log(P(K>=k)) = -(gamma - 1) * log(k) + c
# So: Gamma = 1 - slope
log_x_ccdf = np.log(x_ccdf)
log_y_ccdf = np.log(y_ccdf)

coeffs_ccdf = np.polyfit(log_x_ccdf, log_y_ccdf, 1)
slope_ccdf = coeffs_ccdf[0]
gamma_ccdf = 1 - slope_ccdf

print(f"      Estimated Gamma (CCDF Method): {gamma_ccdf:.2f}")

plt.figure(figsize=(8, 6))
plt.loglog(x_ccdf, y_ccdf, 'b.', markersize=5, alpha=0.5, label='Observed CCDF')
plt.loglog(x_ccdf, np.exp(np.polyval(coeffs_ccdf, log_x_ccdf)), 'r--', linewidth=2,
           label=f'Fit ($\gamma={gamma_ccdf:.2f}$)')
plt.title(f"B. Robust Cumulative Distribution (CCDF)\nGamma ~ {gamma_ccdf:.2f}")
plt.xlabel("Degree (k)")
plt.ylabel("P(K $\geq$ k)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.savefig(os.path.join(PLOT_DIR, "02b_power_law_ccdf_robust.png"))
plt.close()

print("-> Saved: 02b_power_law_pdf_classic.png & 02b_power_law_ccdf_robust.png")

# --- 2.4 CORRELATION ANALYSIS (PEARSON VS SPEARMAN) ---
print("\nTesting the 'Fragility Paradox' (Centrality vs Inefficiency)...")
df_analysis = df_master_nodes.dropna(subset=["avg_delay", "betweenness", "pagerank"]).copy()

if not df_analysis.empty:
    # Pearson checks for linear relationship. Spearman checks for monotonic relationship (rank).
    # Since delays are highly skewed (non-normal), Spearman is the more robust metric here.

    p_corr, p_val = stats.pearsonr(df_analysis["betweenness"], df_analysis["avg_delay"])
    s_corr, s_val = stats.spearmanr(df_analysis["betweenness"], df_analysis["avg_delay"])

    print(f"    - Pearson Correlation:  {p_corr:.4f} (p-value: {p_val:.2e})")
    print(f"    - Spearman Correlation: {s_corr:.4f} (p-value: {s_val:.2e})")

    if s_corr > 0.1:
        conclusion = "CONFIRMED: Central Hubs tend to have higher delays."
    elif s_corr < -0.1:
        conclusion = "REJECTED: Central Hubs are actually more efficient."
    else:
        conclusion = "INCONCLUSIVE: No strong correlation found."
    print(f"Hypothesis Test: {conclusion}")

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x='betweenness', y='avg_delay',
        data=df_analysis,
        scatter_kws={'alpha': 0.5, 'color': 'steelblue'},
        line_kws={'color': 'red', 'label': f'Spearman Trend ({s_corr:.2f})'}
    )

    plt.title(f'Fragility Paradox: Betweenness vs Delay\n(Spearman: {s_corr:.2f}, p<{s_val:.1e})')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Average Departure Delay (min)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "02c_fragility_correlation.png"))
    plt.close()
    print("-> Saved: 02c_fragility_correlation.png")

# --- 2.5 ROBUSTNESS SIMULATION (TARGETED ATTACKS) ---
print("\nRunning Network Disintegration Test...")


def get_giant_component_fraction(g_curr, original_n):
    if len(g_curr) == 0:
        return 0
    gc = max(nx.weakly_connected_components(g_curr), key=len)
    return len(gc) / original_n


fractions = np.linspace(0, 0.20, 11)  # 0% to 20% removal
results = {"Random": [], "Structural": [], "Operational": []}
original_nodes = G.number_of_nodes()

# Define Attack Strategies
nodes_list = list(G.nodes())
random.shuffle(nodes_list)
random_attack = list(nodes_list)

structural_attack = df_master_nodes.sort_values("pagerank", ascending=False)["IATA"].tolist()
structural_attack = [n for n in structural_attack if n in G]

operational_attack = df_master_nodes.sort_values("avg_delay", ascending=False)["IATA"].tolist()
operational_attack = [n for n in operational_attack if n in G]

print("    - Simulating Random, Structural, and Operational failure...")
for f in fractions:
    n_rem = int(original_nodes * f)

    # 1. Random
    G_tmp = G.copy()
    G_tmp.remove_nodes_from(random_attack[:n_rem])
    results["Random"].append(get_giant_component_fraction(G_tmp, original_nodes))

    # 2. Structural (PageRank)
    G_tmp = G.copy()
    G_tmp.remove_nodes_from(structural_attack[:n_rem])
    results["Structural"].append(get_giant_component_fraction(G_tmp, original_nodes))

    # 3. Operational (Delay)
    # Note: operational_attack list is shorter (~570 nodes).
    # If n_rem > len(operational_attack), we assume the attack stops (or we could random fill).
    # Here we simulate strictly targeting the known bad airports.
    G_tmp = G.copy()
    targets = operational_attack[:min(n_rem, len(operational_attack))]
    G_tmp.remove_nodes_from(targets)
    results["Operational"].append(get_giant_component_fraction(G_tmp, original_nodes))

plt.figure(figsize=(10, 6))
plt.plot(fractions, results["Random"], '--', color='gray', label="Random Failure")
plt.plot(fractions, results["Structural"], 'o-', color='firebrick', label="Structural Attack (PageRank)")
plt.plot(fractions, results["Operational"], 's-', color='orange', label="Operational Collapse (High Delay)")

plt.title("Network Robustness: Structural vs Operational Disintegration")
plt.xlabel("Fraction of Nodes Removed")
plt.ylabel("Giant Component Size (Normalized)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "02d_robustness_simulation.png"))
plt.close()
print("-> Saved: 02d_robustness_simulation.png")

# ==============================================================================
# FASE 3: UNSUPERVISED PROFILING ("HEALTH-BASED CLUSTERING")
# ==============================================================================
print("\n" + "="*60)
print("FASE 3: HEALTH-BASED CLUSTERING (K-MEANS + DBSCAN)")
print("="*60)

ckpt_path_fase3 = os.path.join(CHECKPOINT_DIR, "fase3_clusters.pkl")

# --- 3.1 FEATURE ENGINEERING & SCALING ---
print("Preparing Feature Matrix...")

if os.path.exists(ckpt_path_fase2):
    df_clustering = pd.read_pickle(ckpt_path_fase2)
else:
    raise FileNotFoundError("Phase 2 Checkpoint not found. Run Phase 2 first.")

# We select a mix of structural and operational features for clustering
features = ['pagerank', 'betweenness', 'avg_delay', 'delay_variance', 'degree']

# Drop missing values
df_model = df_clustering.dropna(subset=features).copy()
print(f"    - Data points available for clustering: {len(df_model)}")

# Scaling (StandardScaler is crucial because PageRank is ~0.0001 and Delay is ~100.0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[features])

# --- 3.2 HIERARCHICAL CLUSTERING (Exploratory Analysis) ---
print("\nHierarchical Clustering (Dendrogram)...")
# Consider only the top 100 airports by PageRank for the dendrogram to avoid clutter and focus on key nodes
top_100_indices = df_model.sort_values("pagerank", ascending=False).head(100).index
X_sample = df_model.loc[top_100_indices][features]
X_sample_scaled = scaler.transform(X_sample)

plt.figure(figsize=(12, 7))
plt.title("Dendrogram (Top 100 Airports) - Ward Linkage")
dend = shc.dendrogram(shc.linkage(X_sample_scaled, method='ward'))
plt.axhline(y=15, color='r', linestyle='--', label="Suggested Cut")
plt.xlabel("Airport Indices")
plt.ylabel("Euclidean Distances")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "03a_dendrogram_ward.png"))
plt.close()
print("-> Saved: 03a_dendrogram_ward.png")

# --- 3.3 K-MEANS SEGMENTATION & VALIDATION ---
print("\nK-Means Clustering (Grid Search & Validation)...")

k_range = range(2, 7)
results = []

print("    - Validating using Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics...")
print(f"    - Testing k from {min(k_range)} to {max(k_range)}...")
for k in k_range:
    # n_jobs=-1 enables parallel processing
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 1. Silhouette: Cohesion vs Separation (Max is better)
    sil = silhouette_score(X_scaled, labels)
    # 2. Davies-Bouldin: Similarity between clusters (Min is better)
    db = davies_bouldin_score(X_scaled, labels)
    # 3. Calinski-Harabasz: Variance Ratio (Max is better)
    ch = calinski_harabasz_score(X_scaled, labels)

    results.append({"k": k, "Silhouette": sil, "Davies-Bouldin": db, "Calinski-Harabasz": ch})
    print(f"      k={k} | Sil: {sil:.3f} | DB: {db:.3f} | CH: {ch:.1f}")

df_metrics = pd.DataFrame(results)

# Plot Validation Metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.lineplot(x='k', y='Silhouette', data=df_metrics, marker='o', ax=axes[0], color='blue')
axes[0].set_title('Silhouette Score (Higher is better)')

sns.lineplot(x='k', y='Davies-Bouldin', data=df_metrics, marker='o', ax=axes[1], color='red')
axes[1].set_title('Davies-Bouldin Index (Lower is better)')

sns.lineplot(x='k', y='Calinski-Harabasz', data=df_metrics, marker='o', ax=axes[2], color='green')
axes[2].set_title('Calinski-Harabasz Score (Higher is better)')

plt.savefig(os.path.join(PLOT_DIR, "03b_kmeans_validation_metrics.png"))
plt.close()
print("-> Saved: 03b_kmeans_validation_metrics.png")

# --- APPLYING OPTIMAL K-MEANS ---
# ATTENTION: Select best cluster count based on plots and metrics above
best_k = 3
print(f"    -> Applying K-Means with k={best_k}...")
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_model["cluster_kmeans"] = kmeans_final.fit_predict(X_scaled)

# --- 3.4 DBSCAN & OUTLIER DETECTION ---
print("\nDBSCAN for Outlier Detection...")

# 1. k-Distance Graph to find EPS
# ATTENTION: Select min_samples based on expected cluster density
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, min_samples-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title(f"k-Distance Graph (k={min_samples}) for DBSCAN Tuning")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"Epsilon (Dist to {min_samples}-th neighbor)")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "03c_dbscan_kdist_plot.png"))
plt.close()
print("-> Saved: 03c_dbscan_kdist_plot.png")

# 2. Apply DBSCAN
# ATTENTION: Set eps_value based on the "elbow" observed in the k-distance graph above
eps_value = 1.0
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
df_model["cluster_dbscan"] = dbscan.fit_predict(X_scaled)

n_outliers = list(df_model["cluster_dbscan"]).count(-1)
print(f"    -> DBSCAN found {n_outliers} outliers (Noise points labeled as -1)")

# --- 3.5 DIMENSIONALITY REDUCTION & VISUALIZATION ---
print("\nPCA Projection & Profiling...")

# PCA to 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_scaled)
df_model['pca_1'] = principalComponents[:, 0]
df_model['pca_2'] = principalComponents[:, 1]
print(f"    - PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

# Plot K-Means Clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="pca_1", y="pca_2", hue="cluster_kmeans",
    data=df_model, palette="viridis", s=60, alpha=0.8
)
plt.title(f"Airport Clusters (K-Means k={best_k}) on PCA")
plt.savefig(os.path.join(PLOT_DIR, "03d_pca_kmeans_clusters.png"))
plt.close()
print("-> Saved: 03d_pca_kmeans_clusters.png")

# Plot DBSCAN Outliers
plt.figure(figsize=(10, 8))
# Highlight outliers (Cluster -1) in Red, others in Grey
colors = np.where(df_model["cluster_dbscan"] == -1, 'red', 'lightgray')
plt.scatter(df_model["pca_1"], df_model["pca_2"], c=colors, s=60, alpha=0.6)
# Add legend manually
red_patch = mpatches.Patch(color='red', label='Outliers (Noise)')
gray_patch = mpatches.Patch(color='lightgray', label='Normal Data')
plt.legend(handles=[red_patch, gray_patch])
plt.title("DBSCAN Outlier Detection on PCA")
plt.savefig(os.path.join(PLOT_DIR, "03e_pca_dbscan_outliers.png"))
plt.close()
print("-> Saved: 03e_pca_dbscan_outliers.png")

# --- 3.6 PROFILING (SEMANTIC INTERPRETATION) ---
# Boxplots to understand "Who are these clusters?"
print("    - Generating Profiling Boxplots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Boxplot 1: Delay
sns.boxplot(x='cluster_kmeans', y='avg_delay', data=df_model, ax=axes[0], palette="Set2")
axes[0].set_title("Operational Efficiency (Delay)")

# Boxplot 2: PageRank
sns.boxplot(x='cluster_kmeans', y='pagerank', data=df_model, ax=axes[1], palette="Set2")
axes[1].set_yscale('log')
axes[1].set_title("Structural Importance (PageRank)")

# Boxplot 3: Degree
sns.boxplot(x='cluster_kmeans', y='degree', data=df_model, ax=axes[2], palette="Set2")
axes[2].set_yscale('log')
axes[2].set_title("Connectivity (Degree)")

plt.savefig(os.path.join(PLOT_DIR, "03f_cluster_profiling_boxplots.png"))
plt.close()

print("-> Saved: 03f_cluster_profiling_boxplots.png")

# --- EXPORT DELAY REPORT ---
print("Exporting Airport Delay Profile...")
cols_export = [
    "IATA", "Name", "City", "Country",
    "avg_delay", "delay_variance", "num_flights",
    "cluster_kmeans", "degree"
]

df_export = df_model.dropna(subset=["avg_delay"])[cols_export].copy()
df_export = df_export.sort_values("avg_delay", ascending=False)

csv_delay_path = os.path.join(CSV_DIR, "global_airport_delays.csv")
df_export.to_csv(csv_delay_path, index=False)
print(f"-> Delay Report saved to: {csv_delay_path}")

print("\nSaving Checkpoint Phase 3...")
pd.to_pickle(df_model, ckpt_path_fase3)

# ==============================================================================
# FASE 4: RECOMMENDATION SYSTEM (LSH & SMART SWITCH)
# ==============================================================================
print("\n" + "="*60)
print("FASE 4: LSH OPTIMIZATION & RECOMMENDATION ENGINE")
print("="*60)

if os.path.exists(ckpt_path_fase3):
    df_recsys = pd.read_pickle(ckpt_path_fase3)

    # --- CRITICAL FIX: SET IATA AS INDEX ---
    if "IATA" in df_recsys.columns:
        df_recsys = df_recsys.set_index("IATA")
else:
    raise FileNotFoundError("Phase 3 Checkpoint not found.")

# --- 4.1 THEORETICAL TUNING (THE S-CURVE) ---
print("Analyzing LSH parameters (S-Curve)...")

# Parameters for MinHash
num_perm = 128
# Choose bands (b) and rows (r) such that b * r = num_perm
b = 32
r = 4
threshold_theoretical = (1/b)**(1/r)

print(f"    - Configuration: Perm={num_perm}, Bands={b}, Rows={r}")
print(f"    - Theoretical Threshold: {threshold_theoretical:.2f}")

# Plotting the S-Curve
s_values = np.linspace(0, 1, 100)
prob_candidate = 1 - (1 - s_values**r)**b

plt.figure(figsize=(8, 6))
plt.plot(s_values, prob_candidate, color='purple', linewidth=2)
plt.axvline(x=threshold_theoretical, color='r', linestyle='--', label=f'Threshold ~ {threshold_theoretical:.2f}')
plt.title(f"LSH S-Curve (b={b}, r={r})")
plt.xlabel("Jaccard Similarity (s)")
plt.ylabel("Probability of being a Candidate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "04a_lsh_scurve.png"))
plt.close()
print("-> Saved: 04a_lsh_scurve.png")

# --- 4.2 SIGNATURE GENERATION & INDEXING ---
print("\nGenerating MinHash Signatures...")

lsh = MinHashLSH(threshold=threshold_theoretical, num_perm=num_perm)
minhashes = {}
valid_airports = []
sample_node = list(G.nodes())[0]

start_time = time.time()
for n in G.nodes():
    n_str = str(n)

    if n_str in df_recsys.index:
        destinations = list(G.successors(n))
        if destinations:
            m = MinHash(num_perm=num_perm)
            for d in destinations:
                m.update(str(d).encode('utf8'))

            minhashes[n_str] = m
            lsh.insert(n_str, m)
            valid_airports.append(n_str)

print(f"    - Indexed {len(valid_airports)} airports in {time.time() - start_time:.2f} seconds.")

if len(valid_airports) == 0:
    print("\n[ERROR] No airports indexed! Checking mismatch...")
    print(f"    -> Graph Nodes Sample: {list(G.nodes())[:5]}")
    print(f"    -> DataFrame Index Sample: {df_recsys.index[:5].tolist()}")
    raise ValueError("Index Mismatch prevents LSH execution.")

# --- 4.3 EFFICIENCY BENCHMARK (BRUTE FORCE vs LSH) ---
print("\nTesting Computational Efficiency (Brute Force vs LSH)...")

sample_size = 50
if len(valid_airports) < sample_size:
    sample_size = len(valid_airports)

sample_nodes = valid_airports[:sample_size]

# Brute Force
start_bf = time.time()
for n1 in sample_nodes:
    set1 = set(G.successors(n1)) if n1 in G else set()
    for n2 in valid_airports:
        if n1 == n2:
            continue
        set2 = set(G.successors(n2)) if n2 in G else set()

        if len(set1) == 0 or len(set2) == 0:
            continue

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        j_sim = intersection / union
end_bf = time.time()
avg_bf_time = (end_bf - start_bf) / sample_size
print(f"    - Brute Force Avg Time per Query: {avg_bf_time:.4f} seconds")

# LSH Query
start_lsh = time.time()
for n1 in sample_nodes:
    if n1 in minhashes:
        result = lsh.query(minhashes[n1])
end_lsh = time.time()
avg_lsh_time = (end_lsh - start_lsh) / sample_size
print(f"    - LSH Avg Time per Query: {avg_lsh_time:.4f} seconds")

if avg_lsh_time > 0:
    speedup = avg_bf_time / avg_lsh_time
else:
    speedup = 0

print(f"    -> Speedup Factor: {speedup:.1f}x Faster")

plt.figure(figsize=(6, 4))
plt.bar(['Brute Force', 'LSH'], [avg_bf_time, avg_lsh_time], color=['gray', 'green'])
plt.ylabel("Avg Query Time (s)")
plt.yscale('log')
plt.title(f"Efficiency Benchmark (Speedup: {speedup:.0f}x)")
plt.savefig(os.path.join(PLOT_DIR, "04b_benchmark_lsh.png"))
plt.close()
print("-> Saved: 04b_benchmark_lsh.png")

# --- 4.4 SMART SWITCH ENGINE (RECOMMENDATION) ---
print("\nRunning 'Smart Switch' Engine...")

# A. Identify "Bottleneck" Cluster (Highest Avg Delay)
if "cluster_kmeans" in df_recsys.columns:
    cluster_delays = df_recsys.groupby("cluster_kmeans")["avg_delay"].mean()
    bottleneck_cluster = cluster_delays.idxmax()
    bottleneck_airports = df_recsys[df_recsys["cluster_kmeans"] == bottleneck_cluster].index.tolist()

    print(f"    - Target Cluster: #{bottleneck_cluster} (Avg Delay: {cluster_delays[bottleneck_cluster]:.1f} min)")
    print(f"    - Finding alternatives for {len(bottleneck_airports)} bottlenecks...")

    recommendations = []

    for b_node in bottleneck_airports:
        if b_node not in minhashes:
            continue

        # Query LSH for similar airports based on destination overlap
        candidates = lsh.query(minhashes[b_node])

        b_data = df_recsys.loc[b_node]

        for c_node in candidates:
            if c_node == b_node:
                continue

            # Check if candidate exists in our master data
            if c_node not in df_recsys.index:
                continue
            c_data = df_recsys.loc[c_node]

            # --- BUSINESS LOGIC FILTERS ---

            # Filter 1: Delay Improvement
            delay_diff = b_data["avg_delay"] - c_data["avg_delay"]
            if delay_diff <= 5:
                continue    # At least 5 minutes improvement

            # Filter 2: Capacity Constraint
            if c_data["degree"] < (0.5 * b_data["degree"]):
                continue    # At least 50% bottleneck's capacity

            # Calculate Exact Jaccard (Validation)
            set1 = set(G.successors(b_node))
            set2 = set(G.successors(c_node))

            if len(set1.union(set2)) == 0:
                continue
            exact_jaccard = len(set1.intersection(set2)) / len(set1.union(set2))

            recommendations.append({
                "Bottleneck_Airport": b_node,
                "Recommended_Twin": c_node,
                "Similarity (Jaccard)": round(exact_jaccard, 3),
                "Current_Delay": round(b_data["avg_delay"], 1),
                "Expected_Delay": round(c_data["avg_delay"], 1),
                "Potential_Gain (min)": round(delay_diff, 1),
                "Capacity_Match": f"{int(b_data['degree'])} vs {int(c_data['degree'])}"
            })

    # --- 4.5 RESULTS & EXPORT ---
    df_recs = pd.DataFrame(recommendations)

    if not df_recs.empty:
        # Rank by Potential Gain * Similarity (Weighted Score)
        df_recs["Score"] = df_recs["Potential_Gain (min)"] * df_recs["Similarity (Jaccard)"]
        df_recs = df_recs.sort_values("Score", ascending=False)

        print("\n" + "="*30)
        print("TOP 20 SMART RECOMMENDATIONS")
        print("="*30)
        print(df_recs[[
            "Bottleneck_Airport", "Recommended_Twin",
            "Similarity (Jaccard)", "Current_Delay", "Expected_Delay", "Potential_Gain (min)", "Capacity_Match"
        ]].head(20).to_string(index=False))

        # Save to CSV
        csv_path = os.path.join(CSV_DIR, "smart_switch_recommendations.csv")
        df_recs.to_csv(csv_path, index=False)
        print(f"\n    -> Full Report saved to: {csv_path}")

        # Validation Metric: Precision of LSH
        valid_recs = len(df_recs[df_recs["Similarity (Jaccard)"] > 0.3])
        print(f"    -> Quality Check: {valid_recs}/{len(df_recs)} recommendations have High Similarity (>0.3)")

    else:
        print("\n   [!] No valid recommendations found matching all business constraints.")
else:
    print("[ERROR] 'cluster_kmeans' column missing. Did Phase 3 run correctly?")

print("\nPROJECT COMPLETED SUCCESSFULLY")