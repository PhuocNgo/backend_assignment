import re
import pymysql

import readFile
from MFCC import extract_features
import pitch
dir = "D:\Hoc\CSDL DPT\wav"

audio_list = readFile.list_wav_files(dir)

data = []
for audio in audio_list:
    audio_path = dir + "/" + audio
    print(audio_path)

    # Extract MFCC features
    mfcc_features = extract_features(audio_path)

    # Extract pitch features
    pitch_features = pitch.funcPitch_pydub(audio_path)

    # Extract labels from the file name
    match = re.match(r'(\d+-\d+|\d+\+)_([A-Za-z]+)_\d+\.wav', audio)
    if match:
        # Extract range label (e.g., "41-60" or "60+")
        range_label = match.group(1)
        gender_label = match.group(2)  # Extract gender label (e.g., "Nam")
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

# Kết nối tới MySQL
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="fuokqo160802",
    database="csdl_dpt"
)

cursor = connection.cursor()

# Tạo bảng nếu chưa tồn tại
create_table_query = """
CREATE TABLE IF NOT EXISTS output_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    path VARCHAR(255) NOT NULL,
    pitch TEXT NOT NULL,
    MFCCs TEXT NOT NULL,
    range_label VARCHAR(50),
    gender_label VARCHAR(50)
);
"""
cursor.execute(create_table_query)

# Chèn dữ liệu vào bảng
insert_query = """
INSERT INTO output_features (path, pitch, MFCCs, range_label, gender_label)
VALUES (%s, %s, %s, %s, %s)
"""

for path, pitch_feature, mfcc_feature, range_label, gender_label in data:
    cursor.execute(insert_query, (
        path,
        str(pitch_feature),
        str(mfcc_feature.tolist()),
        range_label,
        gender_label
    ))

connection.commit()
cursor.close()
connection.close()

print("Dữ liệu đã được chèn vào MySQL thành công.")
