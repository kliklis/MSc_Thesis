import pandas as pd
import re
import numpy as np

import pandas as pd
import numpy as np

def extract_stats(dataset, column):
    dataset[f'{column}_mean'] = dataset[column].dropna().apply(lambda x: np.mean(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan)
    dataset[f'{column}_std'] = dataset[column].dropna().apply(lambda x: np.std(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan)
    dataset[f'{column}_min'] = dataset[column].dropna().apply(lambda x: np.min(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan)
    dataset[f'{column}_max'] = dataset[column].dropna().apply(lambda x: np.max(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan)
    dataset[f'{column}_count'] = dataset[column].dropna().apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    return dataset

def replace_empty_columns_with_none(df):
    # Iterate over all columns
    for column in df.columns:
        # Check if all values in the column are empty-like
        if df[column].apply(lambda x: str(x).strip() in ["", " ", "nan", "NaN"]).all():
            df[column] = None  # Replace entire column with None

    return df

def convert_Ba_to_A(Ba_file_path):
    # Load the Dataset Ba
    Ba_df = pd.read_csv(Ba_file_path)

    # Debugging: Check available columns
    print("Available columns:", Ba_df.columns)

    # Initialize the converted dataset
    A_columns = [
        "player_id", "clicks", "completion_progress", "session_duration", "escaped", 
        "input_forward", "input_backward", "input_left", "input_right", 
        "riddle_1", "riddle_2", "riddle_3", "riddle_4", "riddle_5", 
        "ommision_errors", "commision_errors", 
        "distraction_1_timestamps", "distraction_2_timestamps", "distraction_3_timestamps", 
        "has_adhd"
    ]
    A_data = []

    for idx, row in Ba_df.iterrows():
        # Extract data for the new structure
        incremental_id = idx + 1  # Incremental ID starting at 1
        ids = row.get('id', '') 
        clicks = row.get('clicks', '')
        
        completion_progress = np.random.uniform(40, 100)  # Assuming random progress
        session_duration = row.get('back', np.nan)  # Use 'back' column for session duration if exists

        escaped = None
        has_adhd = None

        # Directional fields: Use actual values from the corresponding fields
        input_forward = row.get('forward', 0)
        input_backward = row.get('back', 0)
        input_left = row.get('left', 0)
        input_right = row.get('right', 0)

        # Extract riddle completion timestamps and remove { and }
        riddles = [str(row.get(f'riddle_{i}', '')).strip('{}') for i in range(1, 6)]

        # Extract omissions values directly as a comma-separated string
        omissions_values = [
            str(row.get(f'om{i}', 0)) for i in range(1, 7)
        ]
        ommision_errors = ",".join(omissions_values)  # Join values with commas

        # Transfer commissions directly to commision_errors without { and }
        commision_errors = str(row.get('commissions', '')).strip('{}')

        # Distraction timestamps: Remove { and }
        distraction_1 = str(row.get('distractionPot', '')).strip('{}')
        distraction_2 = str(row.get('distractionTimer', '')).strip('{}')
        distraction_3 = str(row.get('distractionWords', '')).strip('{}')

        # Append the processed row to A_data
        A_data.append([
            ids, clicks, completion_progress, session_duration, escaped, 
            input_forward, input_backward, input_left, input_right, 
            riddles[0], riddles[1], riddles[2], riddles[3], riddles[4], 
            ommision_errors, commision_errors, 
            distraction_1, distraction_2, distraction_3, 
            has_adhd
        ])
    
    # Create a DataFrame for Dataset A
    A_df = pd.DataFrame(A_data, columns=A_columns)

    replace_empty_columns_with_none(A_df)

    # Export to CSV
    output_file = Ba_file_path[2:].split('.')[0] + '_Processed.csv'
    A_df.to_csv(output_file, index=False)
    print(f"Processed dataset saved as {output_file}")

# Usage example:
convert_Ba_to_A("./Ba_mentalEscape_gameplaydata.csv")
