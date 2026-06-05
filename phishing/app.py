"""
app.py
------
Flask web application providing the user interface for the
Phishing Email Detection Model.

Users can:
    - Paste any email text into the input box
    - Click "Analyze Email" to get an instant prediction
    - See whether the email is Phishing or Safe
    - View confidence score, URL count, and suspicious keywords found

Run this file AFTER training the model:
    python app.py

Then open: http://127.0.0.1:5000

Author: Phishing Detection Project
"""

import os
from flask import Flask, render_template, request, jsonify
from predictor import PhishingPredictor

# ---------------------------------------------------------------
# Initialize Flask application
# ---------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "phishing_detection_secret_key_2024"

# ---------------------------------------------------------------
# Load the trained predictor once when the server starts
# (avoids reloading the model on every request)
# ---------------------------------------------------------------
predictor = None

def get_predictor():
    """
    Returns the PhishingPredictor instance.
    Creates it lazily on the first request.
    """
    global predictor
    if predictor is None:
        try:
            predictor = PhishingPredictor()
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            predictor = None
    return predictor


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------

@app.route("/")
def index():
    """
    Main page — renders the email detection UI.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint that receives email text and returns prediction.
    Accepts JSON POST requests and returns JSON response.

    Request body:
        {"email_text": "..."}

    Response:
        {
            "success"       : true,
            "label"         : "Phishing Email" or "Safe Email",
            "prediction"    : 1 or 0,
            "confidence"    : 95.3,
            "url_count"     : 2,
            "keyword_count" : 5,
            "found_keywords": ["click", "verify", "urgent"]
        }
    """
    try:
        data       = request.get_json()
        email_text = data.get("email_text", "").strip()

        if not email_text:
            return jsonify({
                "success": False,
                "error": "Please enter some email text to analyze."
            }), 400

        # Get the predictor and run prediction
        pred = get_predictor()

        if pred is None:
            return jsonify({
                "success": False,
                "error": (
                    "Model not found! Please run "
                    "'python train_and_evaluate.py' first."
                )
            }), 500

        result = pred.predict(email_text)
        result["success"] = True

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/sample/<email_type>")
def load_sample(email_type):
    """
    Returns a sample email for demonstration purposes.

    Args:
        email_type (str): "phishing" or "safe"
    """
    samples = {
        "phishing": (
            "URGENT: Your bank account has been suspended due to suspicious activity!\n\n"
            "Dear Customer,\n\n"
            "We have detected unauthorized access to your account. "
            "To avoid permanent suspension, you must verify your identity immediately.\n\n"
            "Click here to login and verify: http://secure-bank-verify-now.xyz\n\n"
            "You must provide your:\n"
            "- Username and password\n"
            "- Credit card number\n"
            "- Social security number\n\n"
            "Failure to verify within 24 hours will result in account closure.\n\n"
            "Security Team\nYour Bank"
        ),
        "safe": (
            "Hi Sarah,\n\n"
            "Hope you are doing well! Just wanted to follow up on our meeting "
            "from last Tuesday regarding the Q3 project timeline.\n\n"
            "Could you please share the updated spreadsheet when you get a chance? "
            "We need to review the milestones before the board meeting on Friday.\n\n"
            "Also, the team lunch is scheduled for Thursday at noon at the Italian "
            "restaurant on Main Street. RSVP by Wednesday if you can make it.\n\n"
            "Thanks a lot!\nJohn"
        )
    }

    sample_text = samples.get(email_type, "")
    return jsonify({"success": True, "text": sample_text})


# ---------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  [SHIELD] PHISHING EMAIL DETECTION WEB APP")
    print("=" * 55)
    print("  Starting server...")
    print("  Open your browser and go to: http://127.0.0.1:5000")
    print("=" * 55 + "\n")

    app.run(debug=True, host="127.0.0.1", port=5000)
