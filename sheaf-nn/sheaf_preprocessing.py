from collections import defaultdict
import random
import pandas as pd
import networkx as nx

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
df = pd.read_csv("././data/UNSW-NB15_4.csv", names=cols)

sfeat = [
    "sport",
    "sttl",
    "Sload",
    "Spkts",
    "smeansz",
    "Sjit",
    "ct_srv_src",
    "ct_src_ltm",
]
dfeat = [
    "dsport",
    "dttl",
    "Dload",
    "Dpkts",
    "dmeansz",
    "Djit",
    "ct_srv_dst",
    "ct_dst_ltm",
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
]


def convert_to_float(value):
    try:
        if isinstance(value, str) and value.startswith("0x"):
            return float(int(value, 16))
        else:
            return float(value)
    except ValueError:
        return None


def check_add_edge(Gtest, src, dst, src_feat, dst_feat, edge_feat):
    if Gtest.has_edge(src, dst):
        Gtest[src][dst]["src_feat"].append(src_feat)
        Gtest[src][dst]["dst_feat"].append(dst_feat)
        Gtest[src][dst]["edge_feat"].append(edge_feat)
    else:
        Gtest.add_edge(
            src, dst, src_feat=[src_feat], dst_feat=[dst_feat], edge_feat=[edge_feat]
        )


def prepare_training_graph(max_edge_density: int, train_size: int):
    df_filtered = df[df["Label"] == 0].head(train_size)
    sampled_df = df_filtered.sample(n=train_size, random_state=42).reset_index(
        drop=True
    )
    df.drop(df_filtered.index, inplace=True)

    Gtrain = nx.DiGraph()
    trainset = {
        "graph": [],
        "srcs": [],
        "dsts": [],
        "sfeats": [],
        "dfeats": [],
        "efeats": [],
    }
    srcs = []
    dsts = []
    srcn = []
    dstn = []
    edges = []

    unique_ips = pd.concat([sampled_df["srcip"], sampled_df["dstip"]]).unique()
    unique_ips = random.sample(
        list(unique_ips), min(round(train_size / max_edge_density), len(unique_ips))
    )

    src_features = sampled_df[sfeat].astype(float)
    sampled_df["dsport"] = sampled_df["dsport"].apply(convert_to_float)
    dst_features = sampled_df[dfeat].astype(float)
    edge_features = sampled_df[efeat]

    edge_features = pd.get_dummies(
        edge_features, columns=["proto", "state", "service"]
    ).astype(float)

    for ip in unique_ips:
        Gtrain.add_node(ip)

    for i, row in sampled_df.iterrows():
        src = row["srcip"]
        dst = row["dstip"]

        src_feat = src_features.loc[i].values
        dst_feat = dst_features.loc[i].values
        edge_feat = edge_features.loc[i].values

        check_add_edge(Gtrain, src, dst, src_feat, dst_feat, edge_feat)
        srcs.append(src)
        dsts.append(dst)
        srcn.append(src_feat)
        dstn.append(dst_feat)
        edges.append(edge_feat)

    trainset["graph"].append(Gtrain)
    trainset["srcs"].append(srcs)
    trainset["dsts"].append(dsts)
    trainset["sfeats"].append(srcn)
    trainset["dfeats"].append(dstn)
    trainset["efeats"].append(edges)

    return trainset


def prepare_test_data(Gtrain, test_size: int, anomaly_rates: list[float]):
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
                "labels": [],
                "attack_type": [],
            }

        srcs = []
        dsts = []
        srcn = []
        dstn = []
        edges = []
        attacks = []
        labels = []

        num_a = round(test_size * ar)
        num_n = test_size - num_a

        df_test = df.copy().reset_index(drop=True)
        df_test["attack_cat"].fillna("None", inplace=True)
        df_test["dsport"] = df_test["dsport"].apply(convert_to_float)
        src_features_df = df_test[sfeat].astype(float)
        dst_features_df = df_test[dfeat].astype(float)
        edge_features_df = df_test[efeat]
        edge_features_df = pd.get_dummies(
            edge_features_df, columns=["proto", "state", "service"]
        ).astype(float)

        for i, row in df_test.iterrows():

            src = row["srcip"]
            dst = row["dstip"]

            src_feat = src_features_df.loc[i].values
            dst_feat = dst_features_df.loc[i].values
            edge_feat = edge_features_df.loc[i].values
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
        testsets[ar]["labels"].append(labels)
        testsets[ar]["attack_type"].append(attacks)
    return testsets


def preprocess(
    sample_size: int, split: float, edge_density: int, anomaly_rates: list[float]
):
    assert 0.0 < split < 1.0
    for ar in anomaly_rates:
        assert 0.0 < ar < 1.0

    test = int(sample_size * split)
    train = int(sample_size * (1 - split))

    trainset = prepare_training_graph(edge_density, train)
    testsets = prepare_test_data(trainset["graph"][0], test, anomaly_rates)

    return trainset, testsets


trainset, testgraphs = preprocess(30000, 0.2, 8, [0.05, 0.1, 0.3])
