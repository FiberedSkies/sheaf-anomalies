from collections import defaultdict
import random
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
import warnings

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

cols = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "Sload",
    "Dload",
    "Spkts",
    "Dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "Sjit",
    "Djit",
    "Stime",
    "Ltime",
    "Sintpkt",
    "Dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "Label",
]

sfeat = [
    "sport",
    "sttl",
    "Sload",
    "sloss",
    "Spkts",
    "smeansz",
    "Sjit",
    "ct_srv_src",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "Stime",
    "Sintpkt",
    "stcpb",
    "swin",
    "sbytes"
]
dfeat = [
    "dsport",
    "dttl",
    "Dload",
    "dloss",
    "Dpkts",
    "dmeansz",
    "Djit",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_dst_sport_ltm",
    "Ltime",
    "Dintpkt",
    "dtcpb",
    "dwin",
    "dbytes"
]
efeat = [
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "service",
    "tcprtt",
    "synack",
    "ackdat",
    "ct_state_ttl",
    "ct_dst_src_ltm",
    "trans_depth",
    "res_bdy_len",
    "ct_ftp_cmd",
]

def convert_to_float(value):
    try:
        if isinstance(value, str) and value.startswith("0x"):
            return float(int(value, 16))
        else:
            return float(value)
    except ValueError:
        return float(8888)
    
    
def record_count(set):
    count = 0
    for edge, features in set.items():
        count += len(features["sfeat"])
    
    return count

def load_and_filter_data(min_records=1000):
    """Load and filter data from all UNSW-NB15 CSV files focusing on well-covered subnet pairs."""
    dfs = []

    for i in range(1, 5):
        file_path = f"././data/UNSW-NB15_{i}.csv"
        print(f"Processing {file_path}")
        df_chunk = pd.read_csv(file_path, names=cols)
        df_chunk['src_subnet'] = df_chunk['srcip'].apply(lambda x: '.'.join(x.split('.')[:3]))
        df_chunk['dst_subnet'] = df_chunk['dstip'].apply(lambda x: '.'.join(x.split('.')[:3]))
        df_chunk['subnet_pair'] = df_chunk.apply(lambda x: f"{x['src_subnet']} to {x['dst_subnet']}", axis=1)
        dfs.append(df_chunk)
    
    full_df = pd.concat(dfs, ignore_index=True)

    subnet_counts = full_df.groupby('subnet_pair').size()
    valid_subnets = subnet_counts[subnet_counts >= min_records].index

    full_df = full_df[full_df['subnet_pair'].isin(valid_subnets)]
    full_df["attack_cat"].fillna("None", inplace=True)
    full_df['ct_ftp_cmd'] = full_df['ct_ftp_cmd'].replace(' ', '0').astype(int)

    full_df['sport'] = full_df['sport'].apply(convert_to_float)
    full_df[sfeat] = full_df[sfeat].astype(float)
    full_df["dsport"] = full_df["dsport"].apply(convert_to_float)
    full_df[dfeat] = full_df[dfeat].astype(float)
    full_df["ct_ftp_cmd"] = full_df["ct_ftp_cmd"].apply(convert_to_float)

    protocol = LabelEncoder()
    state = LabelEncoder()
    service = LabelEncoder()

    full_df["proto"] = protocol.fit_transform(full_df["proto"])
    full_df["state"] = state.fit_transform(full_df["state"])
    full_df["service"] = service.fit_transform(full_df["service"])
    full_df[efeat] = full_df[efeat].astype(float)

    src = RobustScaler()
    dst = RobustScaler()
    e = RobustScaler()
    
    full_df[sfeat] = src.fit_transform(full_df[sfeat])
    full_df[dfeat] = dst.fit_transform(full_df[dfeat])
    full_df[efeat] = e.fit_transform(full_df[efeat])

    src = MinMaxScaler((-1, 1))
    dst = MinMaxScaler((-1, 1))
    e = MinMaxScaler((-1, 1))

    full_df[sfeat] = src.fit_transform(full_df[sfeat])
    full_df[dfeat] = dst.fit_transform(full_df[dfeat])
    full_df[efeat] = e.fit_transform(full_df[efeat])
    del full_df["ct_flw_http_mthd"]
    del full_df["is_ftp_login"]

    target_subnet = "175.45.176 to 149.171.126"
    target_normal = full_df[
        (full_df['subnet_pair'] == target_subnet) & 
        (full_df['Label'] == 0)
    ]

    dominant_subnet = "59.166.0 to 149.171.126"
    if dominant_subnet in valid_subnets:
        dominant_data = full_df[full_df['subnet_pair'] == dominant_subnet]

        edge_counts = dominant_data.groupby(['srcip', 'dstip']).size()
        valid_edges = edge_counts[edge_counts >= 100]
        
        if len(valid_edges) > 0:
            valid_edge_pairs = set(valid_edges.index)
            valid_mask = dominant_data.apply(
                lambda x: (x['srcip'], x['dstip']) in valid_edge_pairs, 
                axis=1
            )

            filtered_dominant = dominant_data[valid_mask]

            if len(filtered_dominant) > 20000:
                filtered_dominant = filtered_dominant.groupby(['srcip', 'dstip']).apply(
                    lambda x: x.sample(
                        n=min(len(x), int(20000 * len(x)/len(filtered_dominant))),
                        random_state=42
                    )
                ).reset_index(drop=True)

            full_df = pd.concat([
                full_df[full_df['subnet_pair'] != dominant_subnet],
                filtered_dominant
            ])
    
    distribution = full_df.groupby('subnet_pair').size()
    subnets_to_drop = distribution[distribution < 1000].index
    full_df = full_df[~full_df['subnet_pair'].isin(subnets_to_drop)]
    distribution = full_df.groupby('subnet_pair').size()
    return full_df, target_normal

def training_set(target_normal: pd.DataFrame, full_df: pd.DataFrame, samples=20000):
    train = {}
    normal = target_normal.head(35349)
    target_normal.drop(normal.index)
    for i, row in normal.iterrows():
        src = row["srcip"]
        dst = row["dstip"]
        edge = src+" to "+dst

        if edge not in train:
            train[edge] = {"sfeat": [], "dfeat": [], "efeat": []}

        sfeats = row[sfeat].values.astype(np.float32)
        dfeats = row[dfeat].values.astype(np.float32)
        efeats = row[efeat].values.astype(np.float32)

        train[edge]["sfeat"].append(sfeats)
        train[edge]["efeat"].append(efeats)
        train[edge]["dfeat"].append(dfeats)
    
    full_df_copy = full_df.copy()
    full_df_copy = full_df_copy[~full_df_copy.index.isin(target_normal.index)]
    full_df_copy["edge"] = full_df_copy["srcip"]+" to "+full_df_copy["dstip"]

    normal_df = full_df_copy[full_df_copy['Label'] == 0]
    edge_counts = normal_df.groupby("edge").size()
    edge_counts = edge_counts[edge_counts >= 200]
    valid_edges = edge_counts[edge_counts <= 2100].index.tolist()
    
    processed = 0
    edges_used = []
    
    random.shuffle(valid_edges)

    for edge in valid_edges:
        if processed > samples:
            break
        edge_records = full_df_copy[full_df_copy["edge"] == edge]
        edges_used.append(edge)

        for i, row in edge_records.iterrows():
            edge = row["edge"]

            if edge not in train:
                train[edge] = {"sfeat": [], "dfeat": [], "efeat": []}

            sfeats = row[sfeat].values.astype(np.float32)
            dfeats = row[dfeat].values.astype(np.float32)
            efeats = row[efeat].values.astype(np.float32)

            train[edge]["sfeat"].append(sfeats)
            train[edge]["efeat"].append(efeats)
            train[edge]["dfeat"].append(dfeats)
        
        processed += len(edge_records)
    
    return train

def test_set(train: dict[dict[float]], target_normal: pd.DataFrame, full_df: pd.DataFrame, testsplit: float, anomalyrate: list[float]):
    tests = defaultdict(dict)
    for ar in anomalyrate:
        count = round(record_count(train) * (testsplit / (1-testsplit)))
        anomalies = round(count * ar)
        normals = round(count * (1 - ar))
        ar_dict = {}
        
        mixrate = 0.2
        tnormal = target_normal.sample(round(normals * (1 - mixrate)), random_state=42)
        full_df["edge"] = full_df["srcip"] + " to " + full_df["dstip"]
        normal_df = full_df[(full_df["Label"] == 0) & (full_df["edge"].isin(train.keys()))]

        records = normal_df.sample(normals - round(normals * (1 - mixrate)), random_state=42)
        attacks =  full_df[(full_df["Label"] == 1) & (full_df["edge"].isin(train.keys()))]
        anoms = attacks.sample(anomalies, random_state=42)
        norm = pd.concat([tnormal, records, anoms])

        for i, row in norm.iterrows():
            src = row["srcip"]
            dst = row["dstip"]
            edge = src+" to "+dst

            if edge not in ar_dict:
                ar_dict[edge] = {"sfeat": [], "dfeat": [], "efeat": [], "label": [], "attack": []}
            
            sfeats = row[sfeat].values.astype(np.float32)
            dfeats = row[dfeat].values.astype(np.float32)
            efeats = row[efeat].values.astype(np.float32)
            labels = row["Label"]
            attacks = row["attack_cat"]

            ar_dict[edge]["sfeat"].append(sfeats)
            ar_dict[edge]["efeat"].append(efeats)
            ar_dict[edge]["dfeat"].append(dfeats)
            ar_dict[edge]["label"].append(labels)
            ar_dict[edge]["attack"].append(attacks)

        tests[ar] = ar_dict
    return tests

def process(anomalyrate: list[float], extra_samples=1000, split=0.15):
    print("[*] Loading and filtering data...")
    full_df, target_normal = load_and_filter_data()
    print("[*] Preparing training dataset...")
    train = training_set(target_normal, full_df, extra_samples)
    print(f"[*] Preparing test sets for anomaly rates {anomalyrate[0]*100}%, {anomalyrate[1]*100}%, and {anomalyrate[2]*100}%")
    tests = test_set(train, target_normal, full_df, split, anomalyrate)
    return train, tests