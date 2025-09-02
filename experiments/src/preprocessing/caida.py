import logging
import pandas as pd
import numpy as np
from experiments.config import settings
import gzip
import shutil
import socket
import dpkt

def compute_caida_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    number_unique_queries = df["src_ip"].nunique()

    stats_per_query = (
        df
        .groupby("src_ip", as_index=False) 
        .agg(Count=("timestamp", "count"))
        .sort_values(by="Count", ascending=False)
        .reset_index(drop=True)  
    )

    stats_per_query.columns = ["src_ip", "Count"]
    stats_per_query.sort_values(by="Count", ascending=False, inplace=True)
    stats_per_query["Rank"] = np.arange(0, len(stats_per_query))

    logging.info(f"Number of unique items (n): {number_unique_queries}")
    return stats_per_query


def preprocess_caida_data(): 
    
    with gzip.open(settings.CAIDA_DATA_COMPRESSED, 'rb') as f_in:
         with open(settings.CAIDA_RAW_DATA, 'wb') as f_out:
             shutil.copyfileobj(f_in, f_out)
    skipped_entries = 0
    error_entries = 0
    data = []
    with open(settings.CAIDA_RAW_DATA, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            try:
                version = buf[0] >> 4
                if version == 4:
                    ip = dpkt.ip.IP(buf)
                    src_ip = socket.inet_ntoa(ip.src)
                elif version == 6:
                    ip6 = dpkt.ip6.IP6(buf)
                    src_ip = socket.inet_ntop(socket.AF_INET6, ip6.src)
                else:
                    skipped_entries += 1
                    continue
                data.append({'src_ip': src_ip, 'timestamp': ts})
            except Exception as e:
                error_entries += 1
    logging.warning(f"Skipped {skipped_entries} entries due to unsupported IP version or parsing errors.")
    logging.warning(f"Encountered {error_entries} errors while processing entries.")       

    df = pd.DataFrame(data)
    start_time = df['timestamp'].min()
    df['AbsMinutes'] = ((df['timestamp'] - start_time) / 60).astype(int)
    ground_truth = compute_caida_ground_truth(df)

    df = df.merge(ground_truth[["src_ip", "Rank"]], on='src_ip', how='left')

    ground_truth.to_csv(settings.CAIDA_GROUND_TRUTH, index=False)
    df.to_csv(settings.CAIDA_CLEANED_DATA, index=False)

    logging.info("CAIDA data preprocessing complete.")