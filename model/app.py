from flask import Flask, request, jsonify
import requests
from func import *


app = Flask(__name__)

LOAD_EMBEDDINGS = True
chat_history = []
initialize_vector_store(LOAD_EMBEDDINGS)


@app.route('/input', methods=['POST'])
def user_input():
    input_text = request.json.get("prompt")
    
    result = chat(input_text, chat_history)
    
    return jsonify({"response": result}), 200

@app.route('/url', methods=['POST'])
def receive_url():
    global LOAD_EMBEDDINGS
    url = request.json.get('url')
    file_name = "downloaded_file.pdf"

    response = requests.get(url)

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
        print(f"Downloaded successfully as {file_name}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

    LOAD_EMBEDDINGS = False
    return "Received", 200


if __name__ == '__main__':
    app.run(debug=True, port=9000)
