
import logging
import pandas as pd
import numpy as np
from experiments.config import settings


def compute_aol_ground_truth(df: pd.DataFrame) -> pd.DataFrame: 
   
    number_unique_queries = df["Query"].nunique()

    stats_per_query = (
        df
        .groupby("Query", as_index=False) 
        .agg(Count=("QueryTime", "count"))
        .sort_values(by="Count", ascending=False)
        .reset_index(drop=True)  
    )

    stats_per_query.columns = ["Query", "Count"]
    stats_per_query.sort_values(by="Count", ascending=False, inplace=True)
    stats_per_query["Rank"] = np.arange(0, len(stats_per_query))

    #logging.info(f"Number of unique items (n): {number_unique_queries}")
    return stats_per_query


def clean_aol_data() -> pd.DataFrame:
    df = pd.read_csv(settings.AOL_RAW_DATA, sep='\t')
    num_original_entries = len(df)

    # Remove duplicates
    df.drop_duplicates(subset=['Query', 'QueryTime'], keep='first', inplace=True)
    
    # Remove entries with empty or missing queries
    df = df[(df['Query'] != "-") & (df['Query'] != "") & df['Query'].notna()]
    df.sort_values(by=['QueryTime'], inplace=True)
    df.reset_index(drop=True, inplace=True)  
    df["QueryTime"] = pd.to_datetime(df['QueryTime'])
    df["Date"] = df["QueryTime"].dt.floor('D')  

    start_date = df["Date"].min()
    df["AbsDay"] = (df["Date"] - start_date).dt.days
    logging.info("Original number of entries: %d", num_original_entries)
    logging.info("Number of entries removed (duplicates and empty queries): %d", num_original_entries-len(df))
    
    return df[["Query", "QueryTime", "Date", "AbsDay"]]
    

def preprocess_aol_data(): 
    
    cleaned_df = clean_aol_data()
    ground_truth = compute_aol_ground_truth(cleaned_df)

    cleaned_df_with_rank = cleaned_df.merge(ground_truth[["Query", "Rank"]], on='Query', how='left')

    ground_truth.to_csv(settings.AOL_GROUND_TRUTH, index=False)
    cleaned_df_with_rank.to_csv(settings.AOL_CLEANED_DATA, index=False)

    logging.info("AOL data preprocessing complete.")
