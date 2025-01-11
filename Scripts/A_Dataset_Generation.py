import pandas as pd
import numpy as np
import random
import CustomUtils


# Generate a dataset for the escape room game
def generate_dataset(
    num_players=1000,
    adhd_ratio=0.15,
    room_difficulty=0.6
):
    CustomUtils.Log("Starting dataset generation...")

    # Initialize list to hold rows
    data_list = []

    for player_id in range(1, num_players + 1):
        if player_id % 100 == 0:
            CustomUtils.Log(f"Generating data for player {player_id}...")

        # Assign ADHD status based on adhd_ratio
        has_adhd = 1 if random.uniform(0, 1) <= adhd_ratio else 0

        # Determine if the player escapes based on room_difficulty
        escaped = 1 if random.uniform(0, 1) <= room_difficulty else 0

        # Generate completion progress
        if escaped:
            completion_progress = 100.0
        else:
            completion_progress = round(random.uniform(20, 99), 2)

        # Generate session duration
        if escaped:
            session_duration = round(
                np.random.normal(600, 120) if not has_adhd else np.random.normal(800, 200),
                2,
            )
        else:
            session_duration = round(random.uniform(300, 600), 2)

        # Generate riddle click timestamps with chronological sequence
        def generate_riddle_timestamps(completion_progress):
            timestamps = []
            time_cursor = 0
            riddles_filled = int(completion_progress / 20)  # Calculate how many riddles are filled (20% per riddle)

            for i in range(1, 6):  # Loop for 5 riddles
                if i <= riddles_filled:  # Fill only if within the completion range
                    riddle_time = sorted(np.random.uniform(time_cursor, session_duration, np.random.randint(3, 6)).tolist())
                    timestamps.append(",".join(map(str, riddle_time)))
                    time_cursor = riddle_time[-1] + random.uniform(10, 30)  # Add buffer time
                else:
                    timestamps.append("")  # Leave empty for riddles beyond progress
            return timestamps

        riddle_1, riddle_2, riddle_3, riddle_4, riddle_5 = generate_riddle_timestamps(completion_progress)

        # Generate distraction timestamps with chronological sequence
        def generate_distraction_timestamps(completion_progress):
            timestamps = []
            time_cursor = 0

            # Determine the number of distractions based on completion progress
            distractions_filled = 0
            if completion_progress > 80:
                distractions_filled = 3
            elif completion_progress > 40:
                distractions_filled = 2
            elif completion_progress > 20:
                distractions_filled = 1

            for i in range(1, 4):  # Generate timestamps for up to 3 distractions
                if i <= distractions_filled:  # Generate only for eligible distractions
                    start = round(random.uniform(time_cursor, session_duration / 2), 2)
                    response = round(start + random.uniform(1, 10), 2)
                    resolution = round(response + random.uniform(1, 10), 2)
                    timestamps.append(f"{start},{response},{resolution}")
                    time_cursor = resolution + random.uniform(10, 30)  # Add buffer time
                else:
                    timestamps.append("")  # Leave remaining distractions empty
            return timestamps

        distraction_1_timestamps, distraction_2_timestamps, distraction_3_timestamps = generate_distraction_timestamps(completion_progress)

        # Generate input times
        input_forward = np.random.randint(10, 500)
        input_backward = np.random.randint(10, 200)
        input_left = np.random.randint(10, 300)
        input_right = np.random.randint(10, 300)

        # Generate errors
        ommision_errors = ",".join(map(str, [np.random.randint(0, 5) for _ in range(6)]))
        commision_errors = ",".join(map(str, [np.random.randint(0, 5) for _ in range(3)]))

        # Generate click value
        clicks = random.randint(3, 100)

        # Append row to data list
        data_list.append(
            {
                "player_id": player_id,
                "clicks": clicks,
                "input_forward": input_forward,
                "input_backward": input_backward,
                "input_left": input_left,
                "input_right": input_right,
                "riddle_1": riddle_1,
                "riddle_2": riddle_2,
                "riddle_3": riddle_3,
                "riddle_4": riddle_4,
                "riddle_5": riddle_5,
                "ommision_errors": ommision_errors,
                "commision_errors": commision_errors,
                "distraction_1_timestamps": distraction_1_timestamps,
                "distraction_2_timestamps": distraction_2_timestamps,
                "distraction_3_timestamps": distraction_3_timestamps,
                "session_duration": session_duration,
                "completion_progress": completion_progress,
                "escaped": escaped,
                "has_adhd": has_adhd,
            }
        )

    # Convert list of dictionaries to a DataFrame
    CustomUtils.Log("Converting data to DataFrame...")
    df = pd.DataFrame(data_list)
    
    CustomUtils.Log("Dataset generation completed!")
    return df

def main():
    # Run the dataset generation
    dataset = generate_dataset()
    
    output_file="../Datasets/A_Labeled.csv"

    CustomUtils.export_dataset(output_file,dataset )

if __name__ == "__main__":
    main()