"""
Flask Image Recognition Application for Assignment 2.
Handles image uploads and performs sign predictions.
"""

from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiate Flask app
app = Flask(__name__)


@app.route("/")
def main():
    """Render the homepage with the image upload form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """
    Handle image upload, processing, and prediction.
    Returns the prediction result or error message.
    """
    try:
        if request.method == "POST":
            img = preprocess_img(request.files["file"].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))

    except Exception:
        return render_template(
            "result.html", err="File cannot be processed or no file uploaded"
        )


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
