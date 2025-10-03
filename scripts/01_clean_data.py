import pandas as pd
import numpy as np
import os

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

def load_data():
    """Load the music survey dataset"""
    data_path = os.path.join(BASE_DIR, "../data/raw/music_survey_data.csv")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded successfully")
        return df
    except FileNotFoundError:
        print(f"❌ Dataset not found at {data_path}")
        print("Please ensure the dataset is in the correct location")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
df = load_data()
df.info()
# Video data not available in dataset
df = df.drop(columns='Q16_Music_guilty_pleasure_Video_or_text') 
df = df.drop(columns='Q17_Video_share_concent')
# No non-null values (Inuit?) 
df = df.drop(columns='FirstNation_23_3')

col_names = df.columns.tolist()
for col in col_names:  # 'rows' should be a list of column names
    print(f"Value counts for column: {col}")
    print(df[col].value_counts())
    print("-" * 40)

#TODO doesn't save to file