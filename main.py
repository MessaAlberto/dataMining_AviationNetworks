import difflib
import glob
import os
import sys
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
from scipy import stats as scipy_stats
from datasketch import MinHash, MinHashLSH

from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler as SklearnScaler

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

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

OPENFLIGHTS_PATH = os.path.join(DATA_DIR, "openflights")
BTS_PATH = os.path.join(DATA_DIR, "bts_usa")
ANAC_PATH = os.path.join(DATA_DIR, "anac_br")
CAA_PATH = os.path.join(DATA_DIR, "caa_uk")

for d in [DATA_DIR, PLOT_DIR, CSV_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

os.chdir(DATA_DIR)

# Config
USE_CHECKPOINTS = False

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("AirportNetworkAnalysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

print("Spark Session Active. UI link usually at port 4040.")

# ==============================================================================
# PHASE 1: DATA ENGINEERING & TRUTH CONSTRUCTION
# ==============================================================================
print("\n" + "="*60)
print("PHASE 1: DATA ENGINEERING & INTEGRATION")
print("="*60)

ckpt_path_phase1 = os.path.join(CHECKPOINT_DIR, "phase1_data.pkl")

if USE_CHECKPOINTS and os.path.exists(ckpt_path_phase1):
    print(f"LOADING CHECKPOINT: {ckpt_path_phase1}")
    df_master_nodes, G = pd.read_pickle(ckpt_path_phase1)
else:
    # --- 1.1 STRUCTURE BUILDING (OpenFlights) ---
    print("Reading raw data with PySpark...")

    df_airports_spark = spark.read.option("header", "true") \
        .option("nullValue", "\\N") \
        .option("inferSchema", "true") \
        .csv(os.path.join(OPENFLIGHTS_PATH, "airports.csv"))

    df_routes_spark = spark.read.option("header", "true") \
        .option("nullValue", "\\N") \
        .option("inferSchema", "true") \
        .csv(os.path.join(OPENFLIGHTS_PATH, "routes.csv"))

    # Filter on IATA (3-letter codes)
    df_nodes = df_airports_spark.filter(F.col("IATA").rlike("^[A-Z]{3}$")) \
        .select("IATA", "Name", "City", "Country", "Latitude", "Longitude") \
        .withColumn("Latitude", F.col("Latitude").cast(DoubleType())) \
        .withColumn("Longitude", F.col("Longitude").cast(DoubleType()))

    # Filter on IATA and direct flights (Stops == 0)
    df_edges = df_routes_spark.filter(F.col("SourceAirport").rlike("^[A-Z]{3}$")) \
        .filter(F.col("DestAirport").rlike("^[A-Z]{3}$")) \
        .filter(F.col("Stops") == 0) \
        .select(F.col("SourceAirport").alias("src"), F.col("DestAirport").alias("dst"))

    n_nodes_init = df_nodes.count()
    n_edges_init = df_edges.count()
    print(f"Raw Input: {n_nodes_init} airports, {n_edges_init} routes.")

    # --- 1.2 TOPOLOGICAL CLEANING ---
    print("\nCleaning Graph Data...")

    # A. Referential Integrity (Removes routes to non-existent airports)
    df_edges = df_edges.join(df_nodes, df_edges.src == df_nodes.IATA, "left_semi")
    df_edges = df_edges.join(df_nodes, df_edges.dst == df_nodes.IATA, "left_semi")

    n_edges_step_a = df_edges.count()
    diff_a = n_edges_init - n_edges_step_a
    if diff_a > 0:
        print(f"    - Referential Integrity: Dropping {diff_a} phantom routes (unknown airports).")

    # B. Self-Loops
    df_edges = df_edges.filter(F.col("src") != F.col("dst"))

    n_edges_step_b = df_edges.count()
    diff_b = n_edges_step_a - n_edges_step_b
    if diff_b > 0:
        print(f"    - Self-Loops: Dropping {diff_b} circular routes (src == dst).")

    # C. Duplicates
    df_edges = df_edges.dropDuplicates(["src", "dst"])

    n_edges_step_c = df_edges.count()
    diff_c = n_edges_step_b - n_edges_step_c
    if diff_c > 0:
        print(f"    - Deduplication: Merged {diff_c} duplicate routes.")

    # D. Isolated Nodes
    active_src = df_edges.select(F.col("src").alias("IATA"))
    active_dst = df_edges.select(F.col("dst").alias("IATA"))
    active_nodes = active_src.union(active_dst).distinct()

    # Keep only active nodes in the final node set (Semi Join)
    df_nodes_final = df_nodes.join(active_nodes, "IATA", "left_semi")

    n_nodes_final = df_nodes_final.count()
    diff_d = n_nodes_init - n_nodes_final
    if diff_d > 0:
        print(f"    - Connectivity: Dropping {diff_d} isolated airports (noise).")

    print(f"Graph Cleaned: {n_nodes_final} nodes, {n_edges_step_c} edges.")
    print("\nMaterializing cleaned data to Pandas for Graph construction...")
    pdf_nodes = df_nodes_final.toPandas()
    pdf_edges = df_edges.toPandas()

    print("\nBuilding Graph G...")
    G = nx.from_pandas_edgelist(pdf_edges, "src", "dst", create_using=nx.DiGraph())
    G.add_nodes_from(pdf_nodes["IATA"])

    node_attr = pdf_nodes.set_index("IATA")[["Name", "City", "Country", "Latitude", "Longitude"]].to_dict('index')
    nx.set_node_attributes(G, node_attr)

    df_master_nodes = pdf_nodes
    valid_nodes_set = set(df_master_nodes["IATA"].unique())     # For quick lookups during data integration
    print(f"    -> Final Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.\n")

    # --- 1.3 OPERATIONS DATA INTEGRATION (Delay) ---
    print("\nLoading & Aligning Performance Data...")

    # A. USA (BTS)
    print("Processing BTS (USA) Data via PySpark...")

    bts_files = glob.glob(os.path.join(BTS_PATH, "*.csv"))
    if not bts_files:
        # Erase error
        raise ValueError(f"No BTS files found in {BTS_PATH}. Check input folder.")

    df_bts_spark = spark.read.option("header", "true") \
                        .option("inferSchema", "true") \
                        .csv(bts_files)

    print(f"    - Filtering cancelled/diverted flights...")
    df_bts_clean = df_bts_spark.filter(
        (F.col("Cancelled") == 0) &
        (F.col("Diverted") == 0)
    ).withColumn("DepDelayMinutes", F.col("DepDelayMinutes").cast("double"))

    # Map-Reduce to compute metrics per airport
    print(f"    - Computing metrics (average delay, variance, flight count) per airport...")
    df_bts_metrics_spark = df_bts_clean.groupBy("Origin").agg(
        F.mean("DepDelayMinutes").alias("avg_delay"),
        F.stddev("DepDelayMinutes").alias("delay_variance"),
        F.count("DepDelayMinutes").alias("num_flights")
    ).withColumnRenamed("Origin", "IATA")

    # .toPandas() because the result is small (one row per airport)
    df_bts_metrics = df_bts_metrics_spark.toPandas()

    valid_bts = df_bts_metrics[df_bts_metrics["IATA"].isin(valid_nodes_set)]
    print(f" -> USA Metrics computed. Nodes: {len(df_bts_metrics)}")
    print(
        f" -> Raw airports: {len(df_bts_metrics)} -> Matched in Graph: {len(valid_bts)} ({len(valid_bts)/len(df_bts_metrics):.1%} of BTS airports)")

    # B. BRAZIL (ANAC)
    print("\nProcessing ANAC (Brazil) Data via PySpark...")

    anac_files = glob.glob(os.path.join(ANAC_PATH, "*.csv"))
    if not anac_files:
        raise ValueError(f"No ANAC files found in {ANAC_PATH}. Check input folder.")

    raw_rdd = spark.read.option("encoding", "ISO-8859-1") \
        .text(anac_files) \
        .rdd.map(lambda row: row[0]) \
        .filter(lambda line: "Atualizado em" not in line)

    # dataframe conversion
    df_anac_spark = spark.read \
        .option("header", "true") \
        .option("sep", ";") \
        .csv(raw_rdd)

    df_anac_spark = df_anac_spark.filter(F.col("Situação Voo") != "Situação Voo")

    # Extract ICAO codes of Brazilian airports
    df_brazil_airports = df_airports_spark \
        .filter(F.col("Country") == "Brazil") \
        .select("ICAO") \
        .dropna()

    brazil_icao_set = [row["ICAO"] for row in df_brazil_airports.collect()]

    # Filter ANAC to keep only departures from Brazil
    df_anac_spark = df_anac_spark.filter(
        F.col("ICAO Aeródromo Origem").isin(brazil_icao_set)
    )

    print(f"Remaining ANAC rows after Brazil-origin filter: {df_anac_spark.count()}")

    # Convert date and filter only REALIZADO flights
    fmt = "yyyy-MM-dd HH:mm:ss"
    df_anac_proc = df_anac_spark \
        .withColumn("t_real", F.try_to_timestamp(F.col("Partida Real"), F.lit(fmt))) \
        .withColumn("t_sched", F.try_to_timestamp(F.col("Partida Prevista"), F.lit(fmt))) \
        .filter(F.col("Situação Voo") == "REALIZADO")

    df_anac_proc = df_anac_proc.dropna(subset=["t_real", "t_sched"])

    # Compute delay in minutes
    df_anac_proc = df_anac_proc.withColumn(
        "delay_min",
        (F.unix_timestamp("t_real") - F.unix_timestamp("t_sched")) / 60
    )
    df_anac_proc = df_anac_proc.withColumn(
        "delay_min",
        F.when(F.col("delay_min") < 0, 0).otherwise(F.col("delay_min"))
    )

    # Aggregate metrics
    df_anac_metrics_spark = df_anac_proc.groupBy("ICAO Aeródromo Origem").agg(
        F.mean("delay_min").alias("avg_delay"),
        F.stddev("delay_min").alias("delay_variance"),
        F.count("delay_min").alias("num_flights")
    )

    # Convert to Pandas for post-processing (mapping ICAO->IATA)
    df_anac_raw = df_anac_metrics_spark.toPandas()

    print("Mapping ICAO to IATA for Brazil data...")
    # Extract columns ICAO and IATA
    mapping_pandas = df_airports_spark.select("ICAO", "IATA").dropna().toPandas()

    # Create a dictionary for mapping ICAO to IATA
    icao_to_iata = mapping_pandas.set_index("ICAO")["IATA"].to_dict()

    # Apply mapping
    df_anac_raw["IATA"] = df_anac_raw["ICAO Aeródromo Origem"].map(icao_to_iata)

    # Drop rows where mapping failed
    df_anac_metrics = df_anac_raw.dropna(subset=["IATA"]).drop(columns=["ICAO Aeródromo Origem"])

    valid_anac = df_anac_metrics[df_anac_metrics["IATA"].isin(valid_nodes_set)]
    print(f" -> Brazil Metrics computed. Nodes: {len(df_anac_metrics)}")
    print(
        f" -> Raw airports: {len(df_anac_metrics)} -> Matched in Graph: {len(valid_anac)} ({len(valid_anac)/len(df_anac_metrics):.1%} of ANAC airports)")

    # C. UK (CAA)
    uk_files = glob.glob(os.path.join(CAA_PATH, "*.csv"))
    df_uk_metrics = pd.DataFrame()

    if not uk_files:
        raise ValueError(f"No CAA files found in {CAA_PATH}. Check input folder.")

    print("\nProcessing CAA (UK) Data via Pandas...")

    if 'df_airports' not in locals():
        # Extract only IATA and Name for mapping (small dataset, can be in-memory)
        df_airports = df_airports_spark.select("IATA", "Name").toPandas()

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

        valid_uk = df_uk_metrics[df_uk_metrics["IATA"].isin(valid_nodes_set)]
        print(f"    Aggregated CAA (UK) metrics: {len(df_uk_metrics)} unique origin airports.")
        print(
            f"-> Raw Airports: {len(df_uk_metrics)} -> Matched in Graph: {len(valid_uk)} ({len(valid_uk)/len(df_uk_metrics):.1%} of UK airports)")

        df_uk_metrics = valid_uk.drop(columns=["tot"])

    # --- 1.4 FINAL MERGE ---
    print("\nMerging Operational Data into Structural Graph...")
    dfs_to_merge = []

    if 'df_bts_metrics' in locals() and df_bts_metrics is not None:
        dfs_to_merge.append(df_bts_metrics)
    if 'df_anac_metrics' in locals() and df_anac_metrics is not None:
        dfs_to_merge.append(df_anac_metrics)
    if 'df_uk_metrics' in locals() and df_uk_metrics is not None:
        dfs_to_merge.append(df_uk_metrics)

    if dfs_to_merge:
        print("Computing data coverage...")
        df_perf_global = pd.concat(dfs_to_merge, ignore_index=True)

        # Aggregation in case of overlaps (e.g. same airport in multiple datasets)
        df_perf_global = df_perf_global.groupby("IATA").mean().reset_index()

        # Merge LEFT: Keep all graph nodes, enrich those with available data
        df_master_nodes = df_master_nodes.merge(df_perf_global, on="IATA", how="left")

        # Verify Node Coverage
        total_nodes = len(df_master_nodes)
        with_data = df_master_nodes["avg_delay"].notna().sum()
        print(f"    - Airport coverage: {with_data}/{total_nodes} nodes have delay data ({with_data/total_nodes:.1%})")

        edges_in_graph = set(G.edges())
        real_routes = set()

        if 'df_bts_clean' in locals():
            routes_bts_spark = df_bts_clean.select("Origin", "Dest").distinct()
            # Return to Pandas (only a few thousand routes, very light)
            routes_bts_pd = routes_bts_spark.toPandas()

            bts_routes = set(zip(routes_bts_pd["Origin"], routes_bts_pd["Dest"]))
            real_routes.update(bts_routes)

        if 'df_anac_proc' in locals():
            # Note: df_anac_proc has original ICAO columns
            routes_anac_spark = df_anac_proc.select("ICAO Aeródromo Origem", "ICAO Aeródromo Destino").distinct()
            routes_anac_pd = routes_anac_spark.toPandas()

            # Robust ICAO->IATA mapping (caching the dict to avoid repeated lookups)
            if 'icao_to_iata' not in locals():
                mapping_pandas = df_airports_spark.select("ICAO", "IATA").dropna().toPandas()
                icao_to_iata = mapping_pandas.set_index("ICAO")["IATA"].to_dict()

            routes_anac_pd["Src_IATA"] = routes_anac_pd["ICAO Aeródromo Origem"].map(icao_to_iata)
            routes_anac_pd["Dst_IATA"] = routes_anac_pd["ICAO Aeródromo Destino"].map(icao_to_iata)

            # Keep only rows where both IATA codes are known
            valid_anac_routes = routes_anac_pd.dropna(subset=["Src_IATA", "Dst_IATA"])
            anac_routes = set(zip(valid_anac_routes["Src_IATA"], valid_anac_routes["Dst_IATA"]))
            real_routes.update(anac_routes)

        # Uk data does not have route-level info, so we skip it for edge verification

        # Calculate Differences and Update Graph
        if real_routes:
            total_real = len(real_routes)
            covered_edges = len([r for r in real_routes if r in edges_in_graph])
            missing_edges_set = real_routes - edges_in_graph
            missing_count = len(missing_edges_set)

            print(f"    - Unique Real Routes Observed: {total_real}")
            print(
                f"    - Matched in OpenFlights Graph: {covered_edges}/{total_real} ({covered_edges/total_real:.1%} of real routes)")
            print(f"    - MISSING EDGES: {missing_count}/{total_real} ({missing_count/total_real:.1%} of real routes)")

            added_count = 0
            if missing_count > 0:
                print(f"        (Note: These are likely new routes created after OpenFlights DB)")
                print("\nTry to add missing routes to the Graph where possible...")

                graph_nodes = set(G.nodes())
                missing_data = []
                counts = {"Both_Nodes_Exist": 0, "Only_Src_Exists": 0, "Only_Dst_Exists": 0, "Neither_Exists": 0}

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
                    f"    - New Routes (src/dst both in Graph): {counts['Both_Nodes_Exist']}/{missing_count} ({counts['Both_Nodes_Exist']/missing_count:.1%} of missing)")
                print(f"     -> SUCCESS: Added {added_count} new routes to the Graph! The network is now up-to-date.")
                print(f"     -> New Graph Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
                print(
                    f"    - Dead Ends (Only Src in Graph):      {counts['Only_Src_Exists']}/{missing_count} ({(counts['Only_Src_Exists'])/missing_count:.1%} of missing)")
                print(
                    f"    - Incoming (Only Dest in Graph):      {counts['Only_Dst_Exists']}/{missing_count} ({counts['Only_Dst_Exists']/missing_count:.1%} of missing)")
                print(
                    f"    - Unknown (Neither in Graph):         {counts['Neither_Exists']}/{missing_count} ({counts['Neither_Exists']/missing_count:.1%} of missing)")

        else:
            print("\nNo raw route data available for edge verification.")

        print("\nDelay data coverage over the structural graph:")
        print(f"    - Nodes with delay data: {with_data}/{total_nodes} ({with_data/total_nodes:.1%})")
        print(
            f"    - Edges with delay data: {covered_edges + added_count}/{G.number_of_edges()} ({(covered_edges + added_count)/G.number_of_edges():.1%})")

        print("\nSaving Checkpoint Phase 1...")
        pd.to_pickle((df_master_nodes, G), ckpt_path_phase1)
    else:
        raise ValueError("No performance data found! Check input folders.")

print("Generating Phase 1 Diagnostic Plots...")
# --- PLOT: DELAY DISTRIBUTION ---
plt.figure(figsize=(10, 6))
sns.histplot(df_master_nodes['avg_delay'].dropna(), bins=50, kde=True)
plt.title('Global Average Delay Distribution')
plt.savefig(os.path.join(PLOT_DIR, "01a_delay_distribution.png"))
plt.close()
print("-> Saved: 01a_delay_distribution.png")

# --- PLOT: GEOGRAPHIC COVERAGE (The "World Map") ---
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

# --- PLOT: VOLUME VS DELAY (Efficiency Check) ---
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
# PHASE 2: NETWORK ANALYSIS & THE "FRAGILITY PARADOX"
# ==============================================================================
print("\n" + "="*60)
print("PHASE 2: TOPOLOGICAL METRICS & ROBUSTNESS SIMULATION")
print("="*60)

ckpt_path_phase2 = os.path.join(CHECKPOINT_DIR, "phase2_metrics.pkl")

if USE_CHECKPOINTS and os.path.exists(ckpt_path_phase2):
    print(f"LOADING CHECKPOINT: {ckpt_path_phase2}")
    df_master_nodes = pd.read_pickle(ckpt_path_phase2)
else:
    # --- 2.1 TOPOLOGICAL METRICS CALCULATION ---
    print("Calculating Network Centrality Measures...")

    gcc_nodes = max(nx.connected_components(G.to_undirected()), key=len)
    G_gcc = G.subgraph(gcc_nodes).copy()

    print("    - Calculating Degrees...")
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())
    total_degree_dict = dict(G.degree())

    df_master_nodes["in_degree"] = df_master_nodes["IATA"].map(in_degree_dict)
    df_master_nodes["out_degree"] = df_master_nodes["IATA"].map(out_degree_dict)
    df_master_nodes["degree"] = df_master_nodes["IATA"].map(total_degree_dict)

    print("    - Calculating PageRank...")
    pagerank_dict = nx.pagerank(G, alpha=0.85)
    df_master_nodes["pagerank"] = df_master_nodes["IATA"].map(pagerank_dict)

    print("    - Calculating Betweenness...")
    betweenness_dict = nx.betweenness_centrality(G, k=500, normalized=True, seed=42)
    df_master_nodes["betweenness"] = df_master_nodes["IATA"].map(betweenness_dict)

    print("    - Calculating Closeness (GCC)...")
    closeness_gcc = nx.closeness_centrality(G_gcc)
    df_master_nodes["closeness"] = df_master_nodes["IATA"].map(closeness_gcc)

    print("    - Calculating Assortativity Variants...")
    r_out_in = nx.degree_assortativity_coefficient(G, 'out', 'in')
    r_out_out = nx.degree_assortativity_coefficient(G, 'out', 'out')
    r_in_in = nx.degree_assortativity_coefficient(G, 'in', 'in')
    r_in_out = nx.degree_assortativity_coefficient(G, 'in', 'out')

    print("    - Calculating ANND...")
    knn_dict = nx.average_neighbor_degree(G)
    df_master_nodes["knn"] = df_master_nodes["IATA"].map(knn_dict)

    df_master_nodes["closeness"] = df_master_nodes["closeness"].fillna(0.0)
    df_master_nodes["betweenness"] = df_master_nodes["betweenness"].fillna(0.0)
    df_master_nodes["pagerank"] = df_master_nodes["pagerank"].fillna(0.0)

    cols_to_analyze = [
        "in_degree", "out_degree", "degree",
        "pagerank", "betweenness", "closeness", "knn"
    ]

    if "avg_delay" in df_master_nodes.columns:
        cols_to_analyze.append("avg_delay")

    print("\nNetwork Centrality Summary:")
    print(f"    - GCC Size:            {len(G_gcc)} nodes ({len(G_gcc)/len(G):.1%} of total)")
    print(f"    - Assortativity (out, in):      {r_out_in:.4f}")
    print(f"    - Assortativity (out, out):     {r_out_out:.4f}")
    print(f"    - Assortativity (in, in):       {r_in_in:.4f}")
    print(f"    - Assortativity (in, out):      {r_in_out:.4f}")

    stats = df_master_nodes[cols_to_analyze].describe().T

    display_map = {
        "in_degree": "In-Degree", "out_degree": "Out-Degree", "degree": "Total Degree",
        "pagerank": "PageRank", "betweenness": "Betweenness", "closeness": "Closeness",
        "knn": "Avg Neighbor Degree", "avg_delay": "Average Delay"
    }

    for col in cols_to_analyze:
        if col in stats.index:
            row = stats.loc[col]
            name = display_map.get(col, col)
            print(f"    - {name:<20} min={row['min']:.4f}, max={row['max']:.4f}, mean={row['mean']:.4f}")

    top_k = 5
    print(f"\nTop {top_k} nodes by PageRank:")
    print(df_master_nodes[["IATA", "pagerank"]].nlargest(top_k, "pagerank").to_string(index=False))

    print(f"\nTop {top_k} nodes by Betweenness:")
    print(df_master_nodes[["IATA", "betweenness"]].nlargest(top_k, "betweenness").to_string(index=False))

    print("\nSaving Checkpoint Phase 2...")
    pd.to_pickle(df_master_nodes, ckpt_path_phase2)

# --- PLOT: SCALE-FREE PDF & CCDF ---
print("\nVerifying Scale-Free Property (PDF vs CCDF)...")
degrees = [d for n, d in G.degree() if d > 0]

# Method A: Classic PDF
degree_counts = pd.Series(degrees).value_counts().sort_index()
x_pdf = degree_counts.index.values
y_pdf = degree_counts.values / sum(degree_counts.values)
log_x_pdf = np.log(x_pdf)
log_y_pdf = np.log(y_pdf)
coeffs_pdf = np.polyfit(log_x_pdf, log_y_pdf, 1)
gamma_pdf = -coeffs_pdf[0]

plt.figure(figsize=(8, 5))
plt.loglog(x_pdf, y_pdf, 'bo', alpha=0.5, label='Observed P(k)')
plt.loglog(x_pdf, np.exp(np.polyval(coeffs_pdf, log_x_pdf)), 'r--', label=f'Fit ($\gamma={gamma_pdf:.2f}$)')
plt.title(f"A. Classic Degree Distribution (PDF)\nGamma ~ {gamma_pdf:.2f}")
plt.xlabel("Degree (k)")
plt.ylabel("P(k)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.savefig(os.path.join(PLOT_DIR, "02a_power_law_pdf_classic.png"))
plt.close()

# Method B: Robust CCDF


def get_ccdf_distribution(degrees):
    degrees_sorted = np.sort(np.array(degrees))
    ccdf = 1 - (np.arange(len(degrees_sorted)) / len(degrees_sorted))
    return degrees_sorted, ccdf


x_ccdf, y_ccdf = get_ccdf_distribution(degrees)
log_x_c = np.log(x_ccdf)
log_y_c = np.log(y_ccdf)
coeffs_c = np.polyfit(log_x_c, log_y_c, 1)
gamma_ccdf = 1 - coeffs_c[0]

plt.figure(figsize=(8, 5))
plt.loglog(x_ccdf, y_ccdf, 'b.', alpha=0.5, label='Observed CCDF')
plt.loglog(x_ccdf, np.exp(np.polyval(coeffs_c, log_x_c)), 'r--', label=f'Fit ($\gamma={gamma_ccdf:.2f}$)')
plt.title(f"B. Robust Cumulative Distribution (CCDF)\nGamma ~ {gamma_ccdf:.2f}")
plt.xlabel("Degree (k)")
plt.ylabel("P(K $\geq$ k)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.savefig(os.path.join(PLOT_DIR, "02a_power_law_ccdf_robust.png"))
plt.close()

# Method C: Rank-Degree Plot (log-log)
degrees_sorted_desc = np.sort(np.array(degrees))[::-1]

# Rank starts from 1
ranks = np.arange(1, len(degrees_sorted_desc) + 1)

log_r = np.log(ranks)
log_k = np.log(degrees_sorted_desc)

# Linear fit in log-log space
coeffs_rank = np.polyfit(log_r, log_k, 1)
slope_rank = coeffs_rank[0]
gamma_rank = 1 - 1/(slope_rank)

plt.figure(figsize=(8, 5))
plt.loglog(ranks, degrees_sorted_desc, 'b.', alpha=0.5, label='Observed Rank-Degree')
plt.loglog(
    ranks,
    np.exp(np.polyval(coeffs_rank, log_r)),
    'r--',
    label=f'Fit (slope={slope_rank:.2f}, gamma={gamma_rank:.2f})'
)

plt.title("C. Rank–Degree Distribution (log–log)")
plt.xlabel("Rank (r)")
plt.ylabel("Degree k(r)")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()

plt.savefig(os.path.join(PLOT_DIR, "02a_power_law_rank_degree.png"))
plt.close()

print("-> Saved: 02a_power_law_pdf_classic.png & 02a_power_law_ccdf_robust.png & 02a_power_law_rank_degree.png")
print(f"Gamma Estimates -> PDF: {gamma_pdf:.2f}, CCDF: {gamma_ccdf:.2f}, Rank-Degree: {gamma_rank:.2f}")

# PLOT: FRAGILITY PARADOX (Centrality vs Inefficiency)
print("\nTesting the 'Fragility Paradox' (Centrality vs Inefficiency)...")

if "avg_delay" in df_master_nodes.columns and "betweenness" in df_master_nodes.columns:
    df_valid = df_master_nodes.dropna(subset=["betweenness", "avg_delay"])

    if not df_valid.empty:
        # Pearson (Pandas)
        pearson_val = df_valid["betweenness"].corr(df_valid["avg_delay"], method="pearson")

        # Spearman (Scipy)
        s_corr, s_val = scipy_stats.spearmanr(df_valid["betweenness"], df_valid["avg_delay"])

        print(f"    - Pearson Correlation (Pandas):  {pearson_val:.4f}")
        print(f"    - Spearman Correlation (Scipy): {s_corr:.4f} (p-value: {s_val:.2e})")

        if s_corr > 0.1:
            conclusion = "CONFIRMED: Central Hubs tend to have higher delays."
        elif s_corr < -0.1:
            conclusion = "REJECTED: Central Hubs are actually more efficient."
        else:
            conclusion = "INCONCLUSIVE: No strong correlation found."
        print(f"Hypothesis Test: {conclusion}")

        # Plot
        plt.figure(figsize=(10, 6))
        sns.regplot(
            x='betweenness', y='avg_delay',
            data=df_valid,
            scatter_kws={'alpha': 0.5, 'color': 'steelblue'},
            line_kws={'color': 'red', 'label': f'Spearman Trend ({s_corr:.2f})'}
        )
        plt.title('Fragility Paradox: Betweenness vs Delay')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('Avg Delay (min)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, "02b_fragility_correlation.png"))
        plt.close()
        print("-> Saved: 02b_fragility_correlation.png")
    else:
        print("Not enough data for correlation analysis.")
else:
    print("Warning: 'avg_delay' or 'betweenness' column missing.")

# PLOT: ROBUSTNESS SIMULATION (Random vs Targeted Attacks)
print("\nRunning Network Disintegration Test...")


def get_giant_component_fraction(g_curr, original_n):
    if len(g_curr) == 0:
        return 0
    gc = max(nx.weakly_connected_components(g_curr), key=len)
    return len(gc) / original_n

fractions = np.linspace(0, 0.20, 11)
results = {"Random": [], "Structural": [], "Operational": []}
original_nodes = G.number_of_nodes()

# Using Spark to sort nodes by centrality and delay for targeted attacks
df_prep = df_master_nodes.fillna({'pagerank': 0.0, 'avg_delay': 0.0})

random_attack = list(G.nodes())
random.shuffle(random_attack)
structural_attack = df_prep.sort_values("pagerank", ascending=False)["IATA"].tolist()
operational_attack = df_prep.sort_values("avg_delay", ascending=False)["IATA"].tolist()

print("    - Simulating Random, Structural, and Operational failure...")
for f in fractions:
    n_rem = int(original_nodes * f)

    def simulate_attack(attack_list):
        G_tmp = G.copy()
        # Remove first n_rem nodes
        targets = [n for n in attack_list[:n_rem] if n in G_tmp]
        G_tmp.remove_nodes_from(targets)
        return get_giant_component_fraction(G_tmp, original_nodes)

    results["Random"].append(simulate_attack(random_attack))
    results["Structural"].append(simulate_attack(structural_attack))
    results["Operational"].append(simulate_attack(operational_attack))

plt.figure(figsize=(10, 6))
plt.plot(fractions, results["Random"], '--', color='gray', label="Random Failure")
plt.plot(fractions, results["Structural"], 'o-', color='firebrick', label="Structural Attack")
plt.plot(fractions, results["Operational"], 's-', color='orange', label="Operational Collapse")
plt.title("Network Robustness: Structural vs Operational Disintegration")
plt.xlabel("Fraction of Nodes Removed")
plt.ylabel("Giant Component Size (Normalized)")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "02c_robustness_simulation.png"))
plt.close()
print("-> Saved: 02c_robustness_simulation.png")

# ==============================================================================
# PHASE 3: CLUSTERING AIRPORTS BASED ON STRUCTURAL & OPERATIONAL FEATURES
# ==============================================================================
print("\n" + "="*60)
print("PHASE 3: HEALTH-BASED CLUSTERING (K-MEANS + DBSCAN)")
print("="*60)

ckpt_path_phase3 = os.path.join(CHECKPOINT_DIR, "phase3_clusters.pkl")

print("Preparing Feature Matrix...")

if os.path.exists(ckpt_path_phase2):
    df_clustering = pd.read_pickle(ckpt_path_phase2)
else:
    raise FileNotFoundError("Phase 2 Checkpoint not found. Run Phase 2 first.")

# --- 3.1 PREPROCESSING & SCALING ---
# Selection of features for clustering (only those with good coverage and relevance)
features_cols = ['pagerank', 'betweenness', 'avg_delay', 'delay_variance', 'degree']

# Drop missing values (ML don't accept Null)
df_model_pd = df_master_nodes.dropna(subset=features_cols).copy()
print(f"    - Data points available for clustering: {len(df_model_pd)}")

scaler = SklearnScaler()
X = df_model_pd[features_cols].values
X_scaled = scaler.fit_transform(X)

# --- 3.2 HIERARCHICAL STRUCTURE (DENDROGRAM) ---
print("\nExploratory Hierarchical Structure...")

plt.figure(figsize=(12, 7))
plt.title("Hierarchical Dendrogram (Ward)")
shc.dendrogram(
    shc.linkage(X_scaled, method='ward'),
    truncate_mode='lastp',
    p=30,
    show_leaf_counts=True
)
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "03a_dendrogram_ward.png"))
plt.close()

print("-> Saved: 03a_dendrogram_ward.png")

# --- 3.3 K-MEANS VALIDATION (GRID SEARCH) ---
print("\nRunning K-Means Grid Search...")
k_range = range(2, 7)
results = []

for k in k_range:
    # Train K-Means
    kmeans = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Metrics
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)

    results.append({
        "k": k,
        "Silhouette": sil,
        "Davies-Bouldin": db,
        "Calinski-Harabasz": ch
    })

    print(f"      k={k} | Sil: {sil:.3f} | DB: {db:.3f} | CH: {ch:.1f}")

df_metrics = pd.DataFrame(results)

# Plotting K-Means Validation Metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.lineplot(x='k', y='Silhouette', data=df_metrics, marker='o', ax=axes[0])
axes[0].set_title('Silhouette Score (Higher is better)')

sns.lineplot(x='k', y='Davies-Bouldin', data=df_metrics, marker='o', ax=axes[1])
axes[1].set_title('Davies-Bouldin (Lower is better)')

sns.lineplot(x='k', y='Calinski-Harabasz', data=df_metrics, marker='o', ax=axes[2])
axes[2].set_title('Calinski-Harabasz (Higher is better)')

plt.savefig(os.path.join(PLOT_DIR, "03b_kmeans_validation_metrics.png"))
plt.close()

print("-> Saved: 03b_kmeans_validation_metrics.png")

# --- 3.4 FINAL K-MEANS ---
best_k = 2  # Manual selection based on dendrogram + metrics
print(f"\nApplying Final K-Means Model with k={best_k}...")

kmeans_final = SklearnKMeans(n_clusters=best_k, random_state=42, n_init=10)
df_model_pd["cluster_kmeans"] = kmeans_final.fit_predict(X_scaled)

# --- 3.5 OUTLIER DETECTION (PCA -> K-DISTANCE -> DBSCAN) ---
print("\nOutlier Detection with DBSCAN...")

# PCA for visualization only
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
df_model_pd["pca_1"] = principal_components[:, 0]
df_model_pd["pca_2"] = principal_components[:, 1]

print(f"    - PCA explained variance ratio: {pca.explained_variance_ratio_}")

# k-distance graph
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, _ = neighbors_fit.kneighbors(X_scaled)

k_distances = np.sort(distances[:, min_samples-1])

plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.title("k-Distance Graph for DBSCAN")
plt.xlabel("Points sorted by distance")
plt.ylabel("Epsilon distance")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "03c_dbscan_kdist_plot.png"))
plt.close()

print("-> Saved: 03c_dbscan_kdist_plot.png")

# Apply DBSCAN
eps_value = 1.0  # Adjust based on k-distance elbow
dbscan = SklearnDBSCAN(eps=eps_value, min_samples=min_samples)
df_model_pd["cluster_dbscan"] = dbscan.fit_predict(X_scaled)

n_outliers = (df_model_pd["cluster_dbscan"] == -1).sum()
print(f"    -> DBSCAN found {n_outliers} outliers")

# Plot DBSCAN result on PCA
plt.figure(figsize=(10, 8))
colors = np.where(df_model_pd["cluster_dbscan"] == -1, 'red', 'lightgray')
plt.scatter(df_model_pd["pca_1"], df_model_pd["pca_2"], c=colors, s=60, alpha=0.6)
plt.title("DBSCAN Outlier Detection (PCA Projection)")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Inliers', markerfacecolor='lightgray', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Outliers', markerfacecolor='red', markersize=10)
])
plt.savefig(os.path.join(PLOT_DIR, "03d_pca_dbscan_outliers.png"))
plt.close()

print("-> Saved: 03d_pca_dbscan_outliers.png")

# --- 3.6 INTERPRETATION ---
print("\nCluster Interpretation...")

cluster_palette = {0: "green", 1: "orange", 2: "blue", 3: "red", 4: "brown"}

# PCA visualization of K-Means clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="pca_1",
    y="pca_2",
    hue="cluster_kmeans",
    data=df_model_pd,
    palette=cluster_palette,
    s=60,
    alpha=0.8
)
plt.title(f"K-Means Clusters (k={best_k}) on PCA Projection")
plt.savefig(os.path.join(PLOT_DIR, "03e_pca_kmeans_clusters.png"))
plt.close()

print("-> Saved: 03e_pca_kmeans_clusters.png")

# Boxplot profiling
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(
    x='cluster_kmeans',
    y='avg_delay',
    data=df_model_pd,
    hue='cluster_kmeans',
    palette=cluster_palette,
    ax=axes[0],
    legend=False
)
axes[0].set_title("Operational Efficiency (Delay)")

sns.boxplot(
    x='cluster_kmeans',
    y='pagerank',
    data=df_model_pd,
    hue='cluster_kmeans',
    palette=cluster_palette,
    ax=axes[1],
    legend=False
)
axes[1].set_yscale('log')
axes[1].set_title("Structural Importance (PageRank)")

sns.boxplot(
    x='cluster_kmeans',
    y='degree',
    data=df_model_pd,
    hue='cluster_kmeans',
    palette=cluster_palette,
    ax=axes[2],
    legend=False
)
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

df_export = df_model_pd.dropna(subset=["avg_delay"])[cols_export].copy()
df_export = df_export.sort_values("avg_delay", ascending=False)

csv_delay_path = os.path.join(CSV_DIR, "global_airport_delays.csv")
df_export.to_csv(csv_delay_path, index=False)
print(f"-> Delay Report saved to: {csv_delay_path}")

print("\nSaving Checkpoint Phase 3...")
pd.to_pickle(df_model_pd, ckpt_path_phase3)

# ==============================================================================
# PHASE 4: RECOMMENDATION SYSTEM (LSH & SMART SWITCH)
# ==============================================================================
print("\n" + "="*60)
print("PHASE 4: LSH OPTIMIZATION & RECOMMENDATION ENGINE")
print("="*60)

if os.path.exists(ckpt_path_phase3):
    df_recsys = pd.read_pickle(ckpt_path_phase3)

    # --- CRITICAL FIX: SET IATA AS INDEX ---
    if "IATA" in df_recsys.columns:
        df_recsys = df_recsys.set_index("IATA")
else:
    raise FileNotFoundError("Phase 3 Checkpoint not found.")

# THEORETICAL TUNING (THE S-CURVE)
print("Analyzing LSH parameters (S-Curve)...")
num_perm = 128
b = 64
r = 2
threshold_theoretical = (1/b)**(1/r)

print(f"    - Configuration: Perm={num_perm}, Bands={b}, Rows={r}")
print(f"    - Theoretical Threshold: {threshold_theoretical:.2f}")

# plot S-curve
s_values = np.linspace(0, 1, 100)
prob_candidate = 1 - (1 - s_values**r)**b

plt.figure(figsize=(8, 6))
plt.plot(s_values, prob_candidate, color='purple', linewidth=2)
plt.axvline(x=threshold_theoretical, color='r', linestyle='--', label=f'Threshold ~ {threshold_theoretical:.2f}')
plt.title(f"LSH S-Curve (b={b}, r={r})")
plt.xlabel("Jaccard Similarity")
plt.ylabel("Probability Candidate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "04a_lsh_scurve.png"))
plt.close()
print("-> Saved: 04a_lsh_scurve.png")

# set IATA as index if present
if "IATA" in df_model_pd.columns:
    df_model_pd = df_model_pd.set_index("IATA")

# SIGNATURE GENERATION & LSH INDEXING
print("\nGenerating MinHash Signatures (Local)...")
lsh = MinHashLSH(threshold=threshold_theoretical, num_perm=num_perm)
minhashes = {}
valid_airports = []

start_time = time.time()
for n in G.nodes():
    n_str = str(n)
    if n_str in df_model_pd.index:
        destinations = list(G.successors(n))
        if destinations:
            m = MinHash(num_perm=num_perm)
            for d in destinations:
                m.update(str(d).encode('utf8'))
            minhashes[n_str] = m
            lsh.insert(n_str, m)
            valid_airports.append(n_str)

print(f"    - Indexed {len(valid_airports)} airports in {time.time() - start_time:.2f} seconds.")

# EFFICIENCY & CORRECTNESS BENCHMARK (Brute Force vs LSH)
print("\nTesting Computational Efficiency & Correctness (Brute Force vs LSH)...")
sample_size = 50
sample_nodes = valid_airports[:min(len(valid_airports), sample_size)]

bf_times = []
lsh_times = []

tp = 0
fp = 0
fn = 0
tn = 0

for n1 in sample_nodes:
    set1 = set(G.successors(n1)) if n1 in G else set()
    if not set1:
        continue

    # Brute Force
    start_bf = time.time()
    bf_candidates = []
    for n2 in valid_airports:
        if n1 == n2:
            continue
        set2 = set(G.successors(n2)) if n2 in G else set()
        if not set2:
            continue
        jac = len(set1 & set2) / len(set1 | set2)
        if jac >= threshold_theoretical:
            bf_candidates.append(n2)
    bf_times.append(time.time() - start_bf)

    # LSH Query
    start_lsh = time.time()
    lsh_candidates = lsh.query(minhashes[n1])
    lsh_times.append(time.time() - start_lsh)

    # Correctness metrics
    bf_set = set(bf_candidates)
    lsh_set = set(lsh_candidates)

    tp += len(bf_set & lsh_set)
    fp += len(lsh_set - bf_set)
    fn += len(bf_set - lsh_set)
    tn += len([n for n in valid_airports if n != n1 and n not in bf_set and n not in lsh_set])

# Aggregate times & efficiency
avg_bf_time = np.mean(bf_times)
avg_lsh_time = np.mean(lsh_times)
speedup = avg_bf_time / avg_lsh_time if avg_lsh_time > 0 else 0

# Metrics
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print(f"    - Brute Force Avg Time: {avg_bf_time:.4f} s")
print(f"    - LSH Avg Time:         {avg_lsh_time:.4f} s")
print(f"    -> Speedup Factor: {speedup:.1f}x Faster\n")

print(f"    --- LSH Correctness Metrics ---")
print(f"    TP: {tp}, TN: {tn}, FN: {fn}, FP: {fp}")
print(f"    Accuracy: {accuracy*100:.2f}%")
print(f"    Precision: {precision*100:.2f}%")
print(f"    Recall:    {recall*100:.2f}%")
print(f"    F1 Score:  {f1*100:.2f}%")

# Plot time comparison
plt.figure(figsize=(6, 4))
plt.bar(['Brute Force', 'LSH'], [avg_bf_time, avg_lsh_time], color=['gray', 'green'])
plt.ylabel("Avg Query Time (s)")
plt.yscale('log')
plt.title(f"Efficiency Benchmark (Speedup: {speedup:.0f}x)")
plt.savefig(os.path.join(PLOT_DIR, "04b_benchmark_lsh.png"))
plt.close()
print("-> Saved: 04b_benchmark_lsh.png")

# Recommendation System
print("\nRunning 'Smart Switch' Engine...")

if "cluster_kmeans" in df_model_pd.columns:
    # Identify bottleneck cluster (highest average delay)
    cluster_delays = df_model_pd.groupby("cluster_kmeans")["avg_delay"].mean()
    bottleneck_cluster = cluster_delays.idxmax()
    bottleneck_airports = df_model_pd[df_model_pd["cluster_kmeans"] == bottleneck_cluster].index.tolist()

    print(f"    - Target Cluster: #{bottleneck_cluster} (Avg Delay: {cluster_delays[bottleneck_cluster]:.1f} min)")
    print(f"    - Finding alternatives for {len(bottleneck_airports)} bottlenecks...")

    recommendations = []

    # Recommendation Loop
    for b_node in bottleneck_airports:
        if b_node not in minhashes:
            continue

        # LSH Query
        candidates = lsh.query(minhashes[b_node])
        b_data = df_model_pd.loc[b_node]

        for c_node in candidates:
            if c_node == b_node or c_node not in df_model_pd.index:
                continue
            c_data = df_model_pd.loc[c_node]

            delay_diff = b_data["avg_delay"] - c_data["avg_delay"]
            if delay_diff <= 5:
                continue  # Minimum gain of 5 minutes
            if c_data["degree"] < (0.5 * b_data["degree"]):
                continue  # Minimum capacity 50%

            # Exact Jaccard calculation for validation
            set1 = set(G.successors(b_node))
            set2 = set(G.successors(c_node))
            if not set1.union(set2):
                continue
            exact_jaccard = len(set1.intersection(set2)) / len(set1.union(set2))

            recommendations.append({
                "Bottleneck_Airport": b_node,
                "Cluster_Match": f"{int(b_data['cluster_kmeans'])} vs {int(c_data['cluster_kmeans'])}",
                "Recommended_Twin": c_node,
                "Similarity": float(exact_jaccard),
                "Current_Delay": float(b_data["avg_delay"]),
                "Expected_Delay": float(c_data["avg_delay"]),
                "Potential_Gain": float(delay_diff),
                "Capacity_Match": f"{int(b_data['degree'])} vs {int(c_data['degree'])}"
            })

    # FINAL ANALYTICS
    if recommendations:
        print("\nAnalyzing Recommendations Score...")
        df_recs = pd.DataFrame(recommendations)

        # Calculate Smart Score
        df_recs["Score"] = df_recs["Potential_Gain"] * df_recs["Similarity"]
        df_recs_sorted = df_recs.sort_values("Score", ascending=False)

        print("\n" + "="*30)
        print("TOP 20 SMART RECOMMENDATIONS (Spark Sorted)")
        print("="*30)

        # Display nicely
        display_cols = ["Bottleneck_Airport", "Recommended_Twin", "Cluster_Match",
                        "Similarity", "Current_Delay", "Potential_Gain", "Capacity_Match"]

        print(df_recs_sorted[display_cols].head(20).to_string(index=False))

        # Stats
        valid_recs = len(df_recs[df_recs["Similarity"] > 0.4])
        print(f"\n    -> Total Recommendations Found: {len(df_recs)}")
        print(f"    -> Quality Check: {valid_recs}/{len(df_recs)} recommendations have High Similarity (>0.4)")
        print(f"    -> Average Potential Gain: {df_recs['Potential_Gain'].mean():.1f} min")
        print(f"    -> Max Potential Gain:     {df_recs['Potential_Gain'].max():.1f} min")
    else:
        print("\n   [!] No valid recommendations found matching all business constraints.")
else:
    print("[ERROR] 'cluster_kmeans' column missing.")

print("\nPROJECT COMPLETED SUCCESSFULLY")
