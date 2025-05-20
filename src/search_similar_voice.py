import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from MFCC import extract_features
from pitch import funcPitch
import ast


def get_feature_vector(audio_path):
    mfcc = extract_features(audio_path)
    pitch = funcPitch(audio_path)
    return np.concatenate([mfcc, pitch]) if mfcc is not None and pitch is not None else None


def estimate_gender(pitch_features):
    mean_pitch = pitch_features[0]  # Mean pitch from funcPitch
    # Heuristic: Male < 165Hz, Female > 165Hz (simplified)
    return "Nam" if mean_pitch < 165 else "Nu"


def search_similar_voices(input_file, db_data):
    try:
        # Convert db_data to DataFrame
        df = pd.DataFrame(db_data)
        if df.empty:
            print("Database is empty.")
            return [], [], []

        # Split database into male and female based on gender_label
        df_male = df[df['gender_label'] == 'Nam'].reset_index(drop=True)
        df_female = df[df['gender_label'] == 'Nu'].reset_index(drop=True)

        # Extract input features and estimate gender
        input_features = get_feature_vector(input_file)
        if input_features is None:
            print("Failed to extract features from input file.")
            return [], [], []

        input_pitch = funcPitch(input_file)
        if input_pitch is None:
            print("Failed to extract pitch from input file.")
            return [], [], []
        input_gender = estimate_gender(input_pitch)

        # Select appropriate database based on input gender
        target_df = df_male if input_gender == "Nam" else df_female

        if target_df.empty:
            print(f"No data available for gender {input_gender} in database.")
            return [], [], []

        # Prepare database features
        db_features = []
        valid_indices = []
        for idx, row in target_df.iterrows():
            try:
                mfcc = np.array(ast.literal_eval(row['MFCCs']))
                pitch = np.array(ast.literal_eval(row['pitch']))
                if mfcc.size == 120 and pitch.size == 4:  # Ensure correct dimensions
                    db_features.append(np.concatenate([mfcc, pitch]))
                    valid_indices.append(idx)
            except (ValueError, SyntaxError):
                continue
        db_features = np.array(db_features)

        if len(db_features) == 0:
            print(
                f"No valid feature vectors for gender {input_gender} in database.")
            return [], [], []

        # Compute similarities
        similarities = cosine_similarity([input_features], db_features)[0]

        # Get top 3
        top_3_indices = np.argsort(similarities)[-3:][::-1]
        top_3_scores = similarities[top_3_indices]
        # Map valid_indices to target_df rows
        top_3_files = target_df.iloc[[valid_indices[i]
                                      for i in top_3_indices]]['path'].values
        top_3_genders = [
            estimate_gender(np.array(ast.literal_eval(
                target_df.iloc[valid_indices[i]]['pitch'])))
            for i in top_3_indices
        ]

        return top_3_files, top_3_scores, top_3_genders

    except Exception as e:
        print(f"Error in search_similar_voices: {e}")
        return [], [], []
