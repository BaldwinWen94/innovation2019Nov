import configparser
import json
import os
import requests
from flask import Flask

app = Flask(__name__)

# load Api Endpoints
config = configparser.ConfigParser()
config.read("endpoints.ini")
endpoint_recycle = config['Endpoints']['recycle']
endpoint_dry = config['Endpoints']['dry']
endpoint_organic = config['Endpoints']['organic']
server_classify = config['Endpoints']['classify']

image_tmp_path = 'image.jpg'


@app.route('/image', methods=['GET'])
def take_image_and_classify():
    take_image(image_tmp_path)
    files = {'image': open(image_tmp_path, 'rb')}
    response = requests.post(server_classify, files=files)
    response = json.loads(response.text)
    return call_recycle_bin_api(response['result'])


def take_image(file_path):
    size = '320*240'
    lightness = '5'
    command = 'fswebcam -r ' + size + ' -S ' + lightness + '--save ' + file_path
    os.system(command)


def call_recycle_bin_api(label):
    endpoint = ""
    response = ""
    if label == 0:
        endpoint = endpoint_dry
        response = "dry"
    elif label == 1:
        endpoint = endpoint_organic
        response = "organic"
    elif label == 2:
        endpoint = endpoint_recycle
        response = "recycle"

    req = requests.get(endpoint)
    if req.status_code == 200:
        print("success call: " + endpoint)
    else:
        print("error call: " + endpoint)

    return response


app.run(host='0.0.0.0', port=5000)
