import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def load_data(file_path, num_hands=None):
    if num_hands is not None:
        total_rows = num_hands
    else:
        total_rows = sum(1 for _ in open(file_path)) - 1  # subtract 1 for header row

    data = pd.read_csv(file_path, nrows=num_hands, chunksize=10000)
    df_list = []
    with tqdm(total=total_rows, unit='hands', desc="Loading data") as pbar:
        for df_chunk in data:
            df_list.append(df_chunk)
            pbar.update(len(df_chunk))

    return pd.concat(df_list, ignore_index=True)

def preprocess_data(data):
    # Convert string representations of lists to actual lists
    data['dealer_final'] = data['dealer_final'].apply(eval)
    data['player_final'] = data['player_final'].apply(eval)
    data['actions_taken'] = data['actions_taken'].apply(eval)  # Convert 'actions_taken' to list
    data['initial_hand'] = data['initial_hand'].apply(eval)  # Convert 'initial_hand' to list
    
    # Create separate rows for split hands
    data = data.explode(['player_final', 'actions_taken'])
    
    # Create 'player_total' column based on 'player_final'
    def calculate_player_total(player_final):
        if isinstance(player_final, list):
            if 'BJ' in player_final:
                return 21
            elif 'Bust' in player_final:
                return 0
            else:
                return player_final[0]
        else:
            return player_final

    data['player_total'] = data['player_final'].apply(calculate_player_total)
    
    # Create 'dealer_total' column based on 'dealer_final'
    def calculate_dealer_total(dealer_final):
        if isinstance(dealer_final, list):
            return dealer_final[-1]
        else:
            return dealer_final

    data['dealer_total'] = data['dealer_final'].apply(calculate_dealer_total)
    
    # Create 'initial_hand_total' column based on 'initial_hand'
    data['initial_hand_total'] = data['initial_hand'].apply(lambda x: sum(x))
    
    # Create 'dealer_up' column based on 'dealer_final'
    data['dealer_up'] = data['dealer_final'].apply(lambda x: x[0] if isinstance(x, list) else x)
    
    # Select the relevant columns for training
    processed_data = data[['dealer_up', 'initial_hand_total', 'dealer_total', 'player_total', 'actions_taken', 'win']]
    
    return processed_data

def split_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data