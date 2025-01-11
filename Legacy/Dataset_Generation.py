import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

# Logging function to print timestamped messages
def Log(message):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] {message}")

# Function to generate a list of random numbers based on the given distribution
def generate_list(size, mean=0.5, std=0.1, distribution='normal'):
    Log(f"Generating list of {size} numbers with {distribution} distribution...")
    if distribution == 'normal':
        numbers = np.random.normal(mean, std, size)
    elif distribution == 'exponential':
        numbers = np.random.exponential(scale=1.0, size=size)
    elif distribution == 'binomial':
        numbers = np.random.binomial(n=10, p=0.5, size=size)
    elif distribution == 'poisson':
        numbers = np.random.poisson(lam=3.0, size=size)
    elif distribution == 'gamma':
        numbers = np.random.gamma(shape=2.0, scale=1.0, size=size)
    elif distribution == 'beta':
        numbers = np.random.beta(a=3.0, b=2.0, size=size)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")
    
    Log(f"Finished generating list of {size} numbers with {distribution} distribution.")
    return numbers.tolist()

def show_histogram(values, bins=20, title='Histogram of Given Values'):
    Log(f"Displaying histogram: {title}")
    plt.figure()
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show(block=False)

# Function to apply thresholding to a list of values
def thresholding(values, threshold):
    Log(f"Applying thresholding at {threshold}")
    return [1 if value > threshold else 0 for value in values]

# Main data generation process
Log("Starting the data generation process...")

# Set constants for number of users and timestamps per user
users_num = 1000
Log(f"Generating data for {users_num} users...")

# Generate random numbers
metrics_random_numbers_normal_distribution = generate_list(users_num, distribution='normal')
metrics_random_numbers_adhd_distribution = generate_list(users_num, distribution='beta')

distractions_random_numbers_normal_distribution = generate_list(users_num, distribution='normal')
distractions_random_numbers_adhd_distribution = generate_list(users_num, distribution='beta')

# Apply thresholding
distractions_random_numbers_normal_distribution_thresholded = thresholding(distractions_random_numbers_normal_distribution, 0.5)
distractions_random_numbers_adhd_distribution_thresholded = thresholding(distractions_random_numbers_adhd_distribution, 0.5)

# Generate column names
Log("Generating column names for the DataFrame...")
columns = ['user_id']
columns.extend([f'_{i}_{metric}' for i in range(1, 7) for metric in [
    'cognitive_flexibility',
    'commission_errors',
    'correct_responses',
    'error_in_answers',
    'inattentiveness',
    'incorrect_clicks',
    'motor_control',
    'ommission_errors',
    'processing_speed',
    'puzzle_duration',
    'reaction_time',
    'response_inhibition',
    'response_speed',
    'sustained_attention',
    'task_switching',
    'time_taken',
    'working_memory']])
columns.extend([f'_{i}_distracted' for i in range(1, 11)])
columns.append('has_adhd')

# Accumulate rows in a list
data_list = []

adhd_positive_ratio = 0.15

# Generate data and populate list of dictionaries
Log("Starting the row generation process...")
for user_id in range(1, users_num + 1):
    if user_id % 100 == 0:
        Log(f"Generating data for user {user_id}...")

    adhd_possibility = random.uniform(0.0, 1.0)
    has_adhd = 1 if adhd_possibility <= adhd_positive_ratio else 0

    # Dictionary to hold the data for the current row
    row_data = {'user_id': user_id}

    # Generate random data for other columns (except user_id and has_adhd)
    for col in columns[1:-11]:
        if has_adhd == 1:
            row_data[col] = random.choice(metrics_random_numbers_adhd_distribution)
        else:
            row_data[col] = random.choice(metrics_random_numbers_normal_distribution)

    for col in columns[-11:-1]:
        if has_adhd == 1:
            row_data[col] = random.choice(distractions_random_numbers_adhd_distribution_thresholded)
        else:
            row_data[col] = random.choice(distractions_random_numbers_normal_distribution_thresholded)

    # Add data for 'has_adhd' column
    row_data['has_adhd'] = has_adhd

    # Append the row_data to the list
    data_list.append(row_data)

# Convert the list of dictionaries to a DataFrame in one go
Log("Converting the data to a DataFrame...")
df = pd.DataFrame(data_list, columns=columns)

# Save the DataFrame to a CSV file for further analysis, without the index column
updated_csv_path = "dataset.csv"
Log(f"Saving the DataFrame to {updated_csv_path}...")
df.to_csv(updated_csv_path, index=False)

Log("Data generation process completed. CSV saved.")
updated_csv_path, df.head()
