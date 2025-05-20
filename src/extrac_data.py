import re
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from read_file import list_wav_files
from MFCC import extract_features
from pitch import funcPitch

dir = "D:/Hoc/CSDL DPT/wav"

# Get list of WAV files
audio_list = list_wav_files(dir)

data = []
for audio in audio_list:
    audio_path = dir + "/" + audio
    print(f"Processing: {audio_path}")

    # Extract MFCC features (120D vector)
    mfcc_features = extract_features(audio_path)

    # Extract pitch features (4D vector)
    pitch_features = funcPitch(audio_path)

    # Skip if feature extraction fails
    if mfcc_features is None or pitch_features is None:
        print(f"Skipping {audio_path}: Failed to extract features")
        continue

    # Extract labels from the file name
    match = re.match(r'(\d+-\d+|\d+\+)_([A-Za-z]+)_\d+\.wav', audio)
    if match:
        range_label = match.group(1)  # e.g., "41-60" or "60+"
        gender_label = match.group(2)  # e.g., "Nam" or "Nu"
    else:
        range_label = "Unknown"
        gender_label = "Unknown"

    data.append((audio_path, pitch_features,
                mfcc_features, range_label, gender_label))

# Prepare data for DataFrame
paths = []
pitches = []
mfccs = []
range_labels = []
gender_labels = []

for path, pitch_feature, mfcc_feature, range_label, gender_label in data:
    paths.append(path)
    # Convert pitch array to JSON string
    pitches.append(json.dumps(pitch_feature.tolist()))
    # Convert MFCC array to JSON string
    mfccs.append(json.dumps(mfcc_feature.tolist()))
    range_labels.append(range_label)
    gender_labels.append(gender_label)

# Create DataFrame
df_new = pd.DataFrame({
    'path': paths,
    'pitch': pitches,
    'MFCCs': mfccs,
    'range_label': range_labels,
    'gender_label': gender_labels
})

# Save to MySQL database
# Replace with your MySQL connection details
mysql_user = "root"
mysql_password = "fuokqo160802"
mysql_host = "localhost"
mysql_database = "csdl_dpt"
table_name = "audio_features"

try:
    # Create SQLAlchemy engine
    engine = create_engine(
        f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_database}")

    # Save DataFrame to MySQL
    df_new.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"Data saved to MySQL table: {table_name}")

except SQLAlchemyError as e:
    print(f"Error saving to MySQL: {str(e)}")
finally:
    # Dispose of the engine to close connections
    if 'engine' in locals():
        engine.dispose()
