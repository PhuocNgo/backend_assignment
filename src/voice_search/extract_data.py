import re
import pymysql
import readFile
from MFCC import extract_features
import pitch
dir = "D:/__pycache__/wav"

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
        range_label = match.group(1)  # Extract range label (e.g., "41-60" or "60+")
        gender_label = match.group(2)  # Extract gender label (e.g., "Nam")
    else:
        range_label = "Unknown"
        gender_label = "Unknown"

    data.append((audio_path, pitch_features, mfcc_features, range_label, gender_label))


# Kết nối tới MySQL
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="csdl_dpt"
)

cursor = connection.cursor()

# Tạo bảng nếu chưa tồn tại
create_table_query = """
CREATE TABLE IF NOT EXISTS output_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    path VARCHAR(255) NOT NULL,
    pitch_1 FLOAT, pitch_2 FLOAT, pitch_3 FLOAT, pitch_4 FLOAT, pitch_5 FLOAT, pitch_6 FLOAT, pitch_7 FLOAT,
    mfcc_1 FLOAT, mfcc_2 FLOAT, mfcc_3 FLOAT, mfcc_4 FLOAT, mfcc_5 FLOAT, mfcc_6 FLOAT, 
    mfcc_7 FLOAT, mfcc_8 FLOAT, mfcc_9 FLOAT, mfcc_10 FLOAT, mfcc_11 FLOAT, mfcc_12 FLOAT, mfcc_13 FLOAT,
    range_label VARCHAR(50),
    gender_label VARCHAR(50)
);
"""
cursor.execute(create_table_query)

# Chèn dữ liệu vào bảng
insert_query = """
INSERT INTO output_features (
    path,
    pitch_1, pitch_2, pitch_3, pitch_4, pitch_5, pitch_6, pitch_7,
    mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7,
    mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13,
    range_label, gender_label
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

for path, pitch_feature, mfcc_feature, range_label, gender_label in data:
    if mfcc_feature is None or pitch_feature is None:
        continue
    cursor.execute(insert_query, (
        path,
        *pitch_feature,
        *mfcc_feature,
        range_label,
        gender_label
    ))

connection.commit()
cursor.close()
connection.close()

print("Dữ liệu đã được chèn vào MySQL thành công.")

# for path, pitch_feature, mfcc_feature, range_label, gender_label in data:
#     paths.append(path)
#     pitches.append(pitch_feature)  # pitch_feature đã là list
#     mfccs.append(mfcc_feature.tolist())  # Convert MFCC array to list
#     range_labels.append(range_label)
#     gender_labels.append(gender_label)

# # Create DataFrame
# df_new = pd.DataFrame({
#     'path': paths,
#     'pitch': pitches,
#     'MFCCs': mfccs,
#     'range_label': range_labels,
#     'gender_label': gender_labels
# })

# # Export DataFrame to .csv file
# output_csv_path = "output_features.csv"
# df_new.to_csv(output_csv_path, index=False)
# print(f"File .csv đã được lưu tại: {output_csv_path}")


