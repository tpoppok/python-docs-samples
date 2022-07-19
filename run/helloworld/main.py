import os
import numpy as np
import json
from PIL import Image
from google.cloud import aiplatform
import flask


app = flask.Flask(__name__, static_url_path="")

@app.get("/")
def prediction():
    IMAGE_DIRECTORY = "cifar_test_images"
    image_files = [file for file in os.listdir(IMAGE_DIRECTORY) if file.endswith(".jpg")]
    image_data = [np.asarray(Image.open(os.path.join(IMAGE_DIRECTORY, file))) for file in image_files]
    x_test = [(image / 255.0).astype(np.float32).tolist() for image in image_data]

    aiplatform.init(project="YOUR_PROJECT_ID", location="ENDPOINT_REGION")
    endpoint = aiplatform.Endpoint("YOUR_ENDPOINT_ID")
    predictions = endpoint.predict(instances=x_test)
    return json.dumps(predictions, indent=2)


if __name__ == "__main__":
    os.environ["FLASK_ENV"] = "development"
    app.run(host="localhost", port=8080, debug=True)