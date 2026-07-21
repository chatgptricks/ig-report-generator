from flask import Flask, request, jsonify
from flask_cors import CORS
from ocr_parser import extract_fields_from_images

app = Flask(__name__)

# Enable CORS for all origins and routes
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "Instagram Report Generator OCR API", "status": "ok"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/extract", methods=["POST", "OPTIONS"])
def extract():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "no images provided"}), 400

        image_bytes_list = [f.read() for f in files]
        result = extract_fields_from_images(image_bytes_list)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

