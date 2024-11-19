"""
A script to preprocess the NB15 Dataset for SNN training
"""

import pandas as pd

# Load Dataset for Processing
path = "C:/Users/alexm/OneDrive/Bureau/School/AMATH 445/project/sheaf_anomalies/data/"
data_src = "UNSW-NB15_4.csv"
cols = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"]
df = pd.read_csv(path + data_src, names = cols)

# Select number of samples to produce
n_samples = 500
random_seed = 42

# Sample and drop a certain subset of the data

df_trunc = df.sample(n_samples, random_state= random_seed).reindex()

# First handle NaN Values

# Count the number of NaN values
print(f"NaN Values in the Sampled Dataset: \n {df_trunc.isna().sum()}")

# The two columns with mostly NaN values are placed as NaN when the applicable protocol is not in use.
# As such, it is reasonable to fill these with 0 values
df_trunc["ct_flw_http_mthd"].fillna(0)
df_trunc["is_ftp_login"].fillna(0)

print("Handled NaN values")

# We use labels for data based on the 

