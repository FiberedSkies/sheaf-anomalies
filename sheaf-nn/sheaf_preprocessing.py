from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
import random
import pandas as pd
import networkx as nx
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
df = pd.read_csv("././data/UNSW-NB15_3.csv", names=cols)

df["attack_cat"].fillna("None", inplace=True)
df['ct_ftp_cmd'] = df['ct_ftp_cmd'].replace(' ', '0').astype(int)
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
        return None

def load_and_filter_data(min_records=2000):
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
    print(f"\nFound {len(valid_subnets)} subnet pairs with {min_records}+ records")

    full_df = full_df[full_df['subnet_pair'].isin(valid_subnets)]

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
    
    print(f"\nFound {len(target_normal)} normal records for target subnet pair")
    print("\nSubnet pair distribution after filtering:")
    distribution = full_df.groupby('subnet_pair').size().sort_values(ascending=False)
    for subnet, count in distribution.items():
        print(f"{subnet}: {count} records")
    
    return full_df, target_normal


def check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat):
    if Gtest.has_edge(src, dst):
        Gtest[src][dst]["src_feat"].append(src_feat)
        Gtest[src][dst]["dst_feat"].append(dst_feat)
        Gtest[src][dst]["edge_feat"].append(edge_feat)
    else:
        Gtest.add_edge(
            src, dst, src_feat=[src_feat], dst_feat=[dst_feat], edge_feat=[edge_feat]
        )


def prepare_training_graph(max_edge_density: int, train_size: int, max_nodes: int, train_normal: pd.DataFrame):
    """Prepare training graph with emphasis on target subnet normal samples and balanced representation."""
    Gtrain = nx.DiGraph()
    trainset = {
        "graph": [],
        "srcs": [],
        "dsts": [],
        "sfeats": [],
        "dfeats": [],
        "efeats": [],
        "enums": [],
    }
    
    srcs, dsts, srcn, dstn, edges, enums = [], [], [], [], [], []

    sampled_df = train_normal.copy()
    edge_counts = sampled_df.groupby(['srcip', 'dstip']).size()
    valid_edges = edge_counts[edge_counts >= 100]
    print(f"Filtered out {len(edge_counts) - len(valid_edges)} edges with fewer than 100 samples")

    valid_edges_df = pd.DataFrame(valid_edges).reset_index()
    valid_edges_df.columns = ['srcip', 'dstip', 'count']
    sampled_df = sampled_df.merge(
        valid_edges_df[['srcip', 'dstip']], 
        on=['srcip', 'dstip'],
        how='inner'
    )

    remaining_size = train_size - len(sampled_df)
    if remaining_size > 0:
        df_filtered = df[df["Label"] == 0]
        df_filtered['src_subnet'] = df_filtered['srcip'].apply(lambda x: '.'.join(x.split('.')[:3]))
        df_filtered['dst_subnet'] = df_filtered['dstip'].apply(lambda x: '.'.join(x.split('.')[:3]))
        df_filtered['subnet_pair'] = df_filtered.apply(lambda x: f"{x['src_subnet']} to {x['dst_subnet']}", axis=1)

        subnet_counts = df_filtered['subnet_pair'].value_counts()
        sampled_dfs = []
        
        for subnet_pair, count in subnet_counts.items():
            if subnet_pair == "175.45.176 to 149.171.126":
                continue
                
            pair_df = df_filtered[df_filtered['subnet_pair'] == subnet_pair]
            sample_size = min(
                250,
                count,
                remaining_size // len(subnet_counts)
            )
            
            if sample_size > 0:
                pair_sample = pair_df.sample(n=sample_size, random_state=42)
                sampled_dfs.append(pair_sample)

        additional_samples = pd.concat(sampled_dfs, ignore_index=True)
        sampled_df = pd.concat([sampled_df, additional_samples], ignore_index=True)

    sampled_df['sport'] = sampled_df['sport'].apply(convert_to_float)
    src_features = sampled_df[sfeat].astype(float)
    sampled_df["dsport"] = sampled_df["dsport"].apply(convert_to_float)
    dst_features = sampled_df[dfeat].astype(float)
    sampled_df["ct_ftp_cmd"] = sampled_df["ct_ftp_cmd"].apply(convert_to_float)
    edge_features = sampled_df[efeat]

    proto_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    service_encoder = LabelEncoder()
    
    edge_features['proto'] = proto_encoder.fit_transform(edge_features['proto'])
    edge_features['state'] = state_encoder.fit_transform(edge_features['state'])
    edge_features['service'] = service_encoder.fit_transform(edge_features['service'])
    edge_features = edge_features.astype(float)

    src_scaler = StandardScaler()
    dst_scaler = StandardScaler()
    edge_scaler = StandardScaler()
    
    src_features = pd.DataFrame(src_scaler.fit_transform(src_features), 
                              index=src_features.index, columns=sfeat)
    dst_features = pd.DataFrame(dst_scaler.fit_transform(dst_features), 
                              index=dst_features.index, columns=dfeat)
    edge_features = pd.DataFrame(edge_scaler.fit_transform(edge_features), 
                               index=edge_features.index, columns=efeat)

    unique_ips = pd.concat([sampled_df["srcip"], sampled_df["dstip"]]).unique()
    unique_ips = random.sample(
        list(unique_ips),
        min(round(train_size / max_edge_density), len(unique_ips))
    )
    unique_ips = unique_ips[:max_nodes]

    for ip in unique_ips:
        Gtrain.add_node(ip)

    for i, row in sampled_df.iterrows():
        src = row["srcip"]
        dst = row["dstip"]
        
        if src not in Gtrain and len(Gtrain.nodes) >= max_nodes:
            continue
        if dst not in Gtrain and len(Gtrain.nodes) >= max_nodes:
            continue
            
        enum = src + " to " + dst
        src_feat = src_features.loc[i].values
        dst_feat = dst_features.loc[i].values
        edge_feat = edge_features.loc[i].values
        
        check_add_edge(Gtrain, src, dst, src_feat, dst_feat, edge_feat)
        srcs.append(src)
        dsts.append(dst)
        srcn.append(src_feat)
        dstn.append(dst_feat)
        edges.append(edge_feat)
        enums.append(enum)

    trainset["graph"].append(Gtrain)
    trainset["srcs"].append(srcs)
    trainset["dsts"].append(dsts)
    trainset["sfeats"].append(srcn)
    trainset["dfeats"].append(dstn)
    trainset["efeats"].append(edges)
    trainset["enums"].append(enums)
    
    print(f"Gtrain has {Gtrain.number_of_nodes()} nodes and {Gtrain.number_of_edges()} edges.")
    print(f"[+] number of training records: {len(edges)}")
    print("\nEdge distribution in training set:")
    edge_counts = Counter(enums)
    for edge, count in edge_counts.most_common():
        print(f"Edge {edge}: {count} training samples")
    
    return trainset



def prepare_test_data(Gtrain, test_size: int, anomaly_rates: list[float]):
    """Prepare test sets from subgraphs of the training graph"""
    testsets = defaultdict(dict)

    for ar in anomaly_rates:
        Gtest = nx.DiGraph()

        if ar not in testsets:
            testsets[ar] = {
                "graph": [],
                "srcs": [],
                "dsts": [],
                "sfeats": [],
                "dfeats": [],
                "efeats": [],
                "enums": [],
                "labels": [],
                "attack_type": [],
            }

        srcs = []
        dsts = []
        srcn = []
        dstn = []
        edges = []
        enums = []
        attacks = []
        labels = []

        num_a = round(test_size * ar)
        num_n = test_size - num_a

        df_test = df.copy().reset_index(drop=True)
        df_test["attack_cat"].fillna("None", inplace=True)
        df_test['sport'] = df_test['sport'].apply(convert_to_float)
        df_test["dsport"] = df_test["dsport"].apply(convert_to_float)
        src_features = df_test[sfeat].astype(float)
        dst_features = df_test[dfeat].astype(float)
        df_test["ct_ftp_cmd"] = df_test["ct_ftp_cmd"].apply(convert_to_float)
        edge_features = df_test[efeat]
        proto_encoder = LabelEncoder()
        state_encoder = LabelEncoder()
        service_encoder = LabelEncoder()
        
        edge_features['proto'] = proto_encoder.fit_transform(edge_features['proto'])
        edge_features['state'] = state_encoder.fit_transform(edge_features['state'])
        edge_features['service'] = service_encoder.fit_transform(edge_features['service'])

        edge_features = edge_features.astype(float)

        src_scaler = StandardScaler()
        dst_scaler = StandardScaler()
        edge_scaler = StandardScaler()
    
        src_features = pd.DataFrame(src_features, columns=sfeat)
        dst_features = pd.DataFrame(dst_features, columns=dfeat)
        edge_features = pd.DataFrame(edge_features, columns=efeat)
        
        src_features_scaled = src_scaler.fit_transform(src_features)
        dst_features_scaled = dst_scaler.fit_transform(dst_features)
        edge_features_scaled = edge_scaler.fit_transform(edge_features)
        
        src_features = pd.DataFrame(src_features_scaled, index=src_features.index, columns=sfeat)
        dst_features = pd.DataFrame(dst_features_scaled, index=dst_features.index, columns=dfeat)
        edge_features = pd.DataFrame(edge_features_scaled, index=edge_features.index, columns=efeat)


        for i, row in df_test.iterrows():

            src = row["srcip"]
            dst = row["dstip"]
            enum = src+" to "+dst

            src_feat = src_features.loc[i].values
            dst_feat = dst_features.loc[i].values
            edge_feat = edge_features.loc[i].values
            label = row["Label"]
            attack = row["attack_cat"]

            if label == 1 and num_a > 0:
                if Gtrain.has_edge(src, dst):
                    if Gtest.has_node(src) and Gtest.has_node(dst):
                        check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat)
                    elif Gtest.has_node(src) and not Gtest.has_node(dst):
                        Gtest.add_node(dst)
                        check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat)
                    elif Gtest.has_node(dst) and not Gtest.has_node(src):
                        Gtest.add_node(src)
                        check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat)
                    else:
                        if src == dst:
                            Gtest.add_node(src)
                        else:
                            Gtest.add_node(src)
                            Gtest.add_node(dst)
                        check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat)

                    srcs.append(src)
                    dsts.append(dst)
                    srcn.append(src_feat)
                    dstn.append(dst_feat)
                    edges.append(edge_feat)
                    attacks.append(attack)
                    labels.append(label)
                    num_a -= 1
                else:
                    continue
            elif label == 0 and num_n > 0:
                if Gtrain.has_edge(src, dst):
                    if Gtest.has_node(src) and Gtest.has_node(dst):
                        check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat)
                    elif Gtest.has_node(src) and not Gtest.has_node(dst):
                        Gtest.add_node(dst)
                        check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat)
                    elif Gtest.has_node(dst) and not Gtest.has_node(src):
                        Gtest.add_node(src)
                        check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat)
                    else:
                        if src == dst:
                            Gtest.add_node(src)
                        else:
                            Gtest.add_node(src)
                            Gtest.add_node(dst)
                        check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat)

                    srcs.append(src)
                    dsts.append(dst)
                    srcn.append(src_feat)
                    dstn.append(dst_feat)
                    edges.append(edge_feat)
                    enums.append(enum)
                    attacks.append(attack)
                    labels.append(label)
                    num_n -= 1
                else:
                    continue
            elif num_n == 0 and num_a == 0:
                break
            else:
                continue

        testsets[ar]["graph"].append(Gtest)
        testsets[ar]["srcs"].append(srcs)
        testsets[ar]["dsts"].append(dsts)
        testsets[ar]["sfeats"].append(srcn)
        testsets[ar]["dfeats"].append(dstn)
        testsets[ar]["efeats"].append(edges)
        testsets[ar]["enums"].append(enums)
        testsets[ar]["labels"].append(labels)
        testsets[ar]["attack_type"].append(attacks)
        print(f"[+] anomaly rate {ar*100}%, number of test records: {len(labels)}")
    return testsets


def preprocess(sample_size: int, split: float, edge_density: int, anomaly_rates: list[float]):
    """Modified preprocessing pipeline"""
    assert 0.0 < split < 1.0
    for ar in anomaly_rates:
        assert 0.0 < ar < 1.0

    filtered_df, target_normal = load_and_filter_data(min_records=1000)

    train_normal = target_normal.sample(n=min(38000, len(target_normal)), random_state=42)

    filtered_df = filtered_df.drop(train_normal.index)

    global df
    df = filtered_df
    
    test = int(sample_size * split)
    train = int(sample_size * (1 - split))

    trainset = prepare_training_graph(edge_density, train, 47, train_normal)
    testsets = prepare_test_data(trainset["graph"][0], test, anomaly_rates)

    return trainset, testsets

def analyze_subnet_anomalies():
    df['src_subnet'] = df['srcip'].apply(lambda x: '.'.join(x.split('.')[:3]))
    df['dst_subnet'] = df['dstip'].apply(lambda x: '.'.join(x.split('.')[:3]))
    df['subnet_pair'] = df.apply(lambda x: f"{x['src_subnet']} to {x['dst_subnet']}", axis=1)
    
    subnet_stats = []
    for subnet_pair, group in df.groupby('subnet_pair'):
        stats = {
            'subnet_pair': subnet_pair,
            'total_samples': len(group),
            'normal_samples': len(group[group['Label'] == 0]),
            'anomaly_samples': len(group[group['Label'] == 1]),
            'anomaly_types': group[group['Label'] == 1]['attack_cat'].value_counts().to_dict()
        }
        subnet_stats.append(stats)
    
    stats_df = pd.DataFrame(subnet_stats)
    stats_df['anomaly_rate'] = stats_df['anomaly_samples'] / stats_df['total_samples']
    
    return stats_df.sort_values('total_samples', ascending=False)
