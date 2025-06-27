from flask import Flask, request, jsonify
import os
from search_similar_voice import search_similar_voices
from flask_cors import CORS
from flask_mysqldb import MySQL

app = Flask(__name__)
CORS(app)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'fuokqo160802'
app.config['MYSQL_DB'] = 'csdl_dpt'
mysql = MySQL(app)


@app.route('/api/search-voices', methods=['POST'])
def search_voices():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        print("audio file:", audio_file)
        temp_path = os.path.join("temp", audio_file.filename)
        os.makedirs("temp", exist_ok=True)
        audio_file.save(temp_path)

        # Fetch data from MySQL
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT path, MFCCs, pitch, gender_label FROM output_features")
        db_data = cur.fetchall()
        cur.close()

        # Convert fetched data to list of dictionaries
        db_data = [
            {"path": row[0], "MFCCs": row[1],

                "pitch": row[2], "gender_label": row[3]}
            for row in db_data
        ]

        top_files, top_scores, top_genders = search_similar_voices(
            temp_path, db_data)
        print("temp_path:", temp_path)
        os.remove(temp_path)  # Clean up

        results = [
            {"file": file, "score": float(score), "gender": gender}
            for file, score, gender in zip(top_files, top_scores, top_genders)
        ]

        print("results:", results)
        return jsonify({"results": results})
    except Exception as e:
        print(f'error: {e}')
        return jsonify({"error": str(e)}), 500


@app.route('/api/test', methods=['GET'])
def hello_world():
    print("Hello world!")
    return jsonify({"message": "Hello World"})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
