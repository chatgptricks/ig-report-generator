from flask import Flask, request, jsonify
from flask_cors import CORS
from ocr_parser import extract_fields_from_images

app = Flask(__name__)

# Lock this down to your GitHub Pages origin in production if desired:
# CORS(app, origins=["https://chatgptricks.github.io"])
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/extract", methods=["POST"])
def extract():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no images provided"}), 400

    image_bytes_list = [f.read() for f in files]
    result = extract_fields_from_images(image_bytes_list)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
