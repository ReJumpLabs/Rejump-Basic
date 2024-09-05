import json
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER
import concurrent.futures

app = Flask(__name__)


def update_single_token(token):
    """Helper function to update data for a single token."""
    print(f"Updating data for {token}...")
    try:
        # Download data for the current token
        files = download_data(token, TRAINING_DAYS, REGION, DATA_PROVIDER)
        
        # Format the downloaded data
        format_data(files, DATA_PROVIDER)
        
        # Train the model for the current token
        train_model(TIMEFRAME)
        
        print(f"Data update and model training completed for {token}")
    except Exception as e:
        print(f"Error updating data for {token}: {str(e)}")


def update_data():
    """Download price data, format data, and train the model for ETH, BNB, and ARB concurrently."""
    tokens_to_update = ["ETH", "BNB", "ARB"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Execute the update_single_token function concurrently for each token
        executor.map(update_single_token, tokens_to_update)


@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
