from flask import Flask, request, jsonify
import requests
app = Flask(__name__)

@app.route('/url', methods=['POST'])
def receive_url():
    url = request.json.get('url')
    file_name = "downloaded_file.pdf"

    response = requests.get(url)

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
        print(f"Downloaded successfully as {file_name}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

    return "Received", 200

@app.get('/ok')
def sayHi():
    return jsonify({ "greet": "helo world" })

if __name__ == '__main__':
    app.run(debug=True, port=9000)
