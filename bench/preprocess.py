"""
A script to preprocess the NB15 Dataset for SNN training
"""

import pandas as pd

# Load Dataset for Processing
path = "././data/"
data_src = "UNSW-NB15_4.csv"
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
df = pd.read_csv(path + data_src, names=cols)

# Select number of samples to produce
n_samples = 36000
random_seed = 42

# Sample and drop a certain subset of the data

df_trunc = df.sample(n_samples, random_state=random_seed)
df_trunc.reset_index(inplace=True, drop=True)

# First handle NaN Values

# Count the number of NaN values
print(f"NaN Values in the Sampled Dataset: \n {df_trunc.isna().sum()}")

# The two columns with mostly NaN values are placed as NaN when the applicable protocol is not in use.
# As such, it is reasonable to fill these with 0 values
df_trunc["ct_flw_http_mthd"].fillna(0, inplace=True)
df_trunc["is_ftp_login"].fillna(0, inplace=True)

# Fill attack type with "None" if no attack

df_trunc["attack_cat"].fillna("None", inplace=True)

print("Handled NaN values")

# The ct_ftp_cmd column has strings instead of 0 for some reason. Handle that separately
df_trunc.loc[df_trunc["ct_ftp_cmd"] == " ", "ct_ftp_cmd"] = 0

# We do not use the labels given by ports, only IP
df_trunc.drop(labels=["sport", "dsport"], axis=1, inplace=True)

# We use labels for data based on the source and destination IP address to create a graph structure

edge_labels = df_trunc["srcip"] + "-to-" + df_trunc["dstip"]

df_trunc["Edge"] = edge_labels

# Sort data entries by edge label

df_trunc.sort_values("Edge", inplace=True)
df_trunc.reset_index(inplace=True, drop=True)

# Split into data, label and categories

graph_labels = ["Edge", "srcip", "dstip"]

timestamp_labels = ["Stime", "Ltime"]

category_label = ["attack_cat"]

target_label = ["Label"]


# Create new dataframes for each type of information
df_data = df_trunc.drop(
    category_label + graph_labels + target_label + timestamp_labels, axis=1
)

df_graph = df_trunc[graph_labels]

df_category = df_trunc[category_label]

df_label = df_trunc[target_label]

df_times = df_trunc[timestamp_labels]

# Encode categorical data
cat_cols = ["proto", "state", "service"]


df_num = df_data.drop(labels=cat_cols, axis=1)
df_cat = df_data[cat_cols]

df_cat = pd.get_dummies(df_cat)

df_data = pd.concat([df_num, df_cat], axis=1)


# Save results

df_data.to_csv(path + "dataset.csv")
df_graph.to_csv(path + "graph.csv")
df_category.to_csv(path + "attack.csv")
df_label.to_csv(path + "label.csv")
df_times.to_csv(path + "timestamps.csv")
