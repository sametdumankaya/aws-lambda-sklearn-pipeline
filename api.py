from flask import Flask, request, json
import boto3
import pickle

BUCKET_NAME = 'text-classification-pipeline'
MODEL_FILE_NAME = 'pipeline2.pkl'
app = Flask(__name__)
S3 = boto3.client('s3', region_name='eu-central-1')


def load_model(key):
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    model_str = response['Body'].read()
    model = pickle.loads(model_str)
    return model


@app.route('/', methods=['POST'])
def index():
    body_dict = request.get_json(silent=True)
    data = body_dict['data']
    model = load_model(MODEL_FILE_NAME)
    prediction = model.predict([data]).tolist()
    result = {'prediction': prediction}

    return json.dumps(result)


if __name__ == '__main__':
    # listen on all IPs
    app.run(host='0.0.0.0')
