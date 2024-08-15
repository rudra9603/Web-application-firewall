from flask import Flask, request, jsonify
import numpy as np
import pickle
import urllib.parse

# Load the trained model
with open('finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

badwords = ['sleep', 'uid', 'select', 'waitfor', 'delay', 'system', 'union', 'order by', 'group by', 'admin', 'drop', 'script']

def ExtractFeatures(path, body):
    path = str(path)
    body = str(body)
    combined_raw = path + body
    raw_percentages = combined_raw.count("%")
    raw_spaces = combined_raw.count(" ")

    # Check if both counts exceed the threshold
    raw_percentages_count = raw_percentages if raw_percentages > 3 else 0
    raw_spaces_count = raw_spaces if raw_spaces > 3 else 0

    # Decode the path and body for other feature extractions
    path_decoded = urllib.parse.unquote_plus(path)
    body_decoded = urllib.parse.unquote_plus(body)

    single_q = path_decoded.count("'") + body_decoded.count("'")
    double_q = path_decoded.count("\"") + body_decoded.count("\"")
    dashes = path_decoded.count("--") + body_decoded.count("--")
    braces = path_decoded.count("(") + body_decoded.count("(")
    spaces = path_decoded.count(" ") + body_decoded.count(" ")
    semicolons = path_decoded.count(";") + body_decoded.count(";")
    angle_brackets = path_decoded.count("<") + path_decoded.count(">") + body_decoded.count("<") + body_decoded.count(">")
    special_chars = sum(path_decoded.count(c) + body_decoded.count(c) for c in '$&|')

    badwords_count = sum(path_decoded.lower().count(word) + body_decoded.lower().count(word) for word in badwords)

    path_length = len(path_decoded)
    body_length = len(body_decoded)

    return [single_q, double_q, dashes, braces, spaces, raw_percentages_count, semicolons, angle_brackets, special_chars, path_length, body_length, badwords_count]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    path = data.get('path', '')
    body = data.get('body', '')

    features = ExtractFeatures(path, body)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    response = {
        'prediction': int(prediction[0]),
        'message': 'Intrusion Detected!' if prediction[0] == 1 else 'No Intrusion'
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
