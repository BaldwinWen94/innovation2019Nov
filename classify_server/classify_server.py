import os

import flask
import numpy as np
import torch
import werkzeug
from PIL import Image
from flask import Flask
from torchvision.transforms import transforms
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load PyTorch Model
model = torch.load("model/waste_classify.pt", map_location='cpu')
model.eval()

# define PyTorch variables
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
transformations = transforms.Compose([
    transforms.RandomResizedCrop(size=256),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])


@app.route('/classify', methods=['POST'])
def classify_image():
    image_file = flask.request.files['image']
    filename = "test/" + werkzeug.utils.secure_filename(image_file.filename)
    print("\nReceived image File name : " + image_file.filename)
    image_file.save(filename)
    top_label = classify_image(filename)
    return {"result": top_label}


photo_url = "../../Downloads/test1.jpg"
test_url = "test/test1.jpg"


@app.route('/auto_classify', methods=['GET'])
def auto_classify_image():
    image = Image.open(photo_url)
    rgb_im = image.convert('RGB')
    rgb_im.save(test_url)
    top_label = classify_image(test_url)
    delete_file(photo_url)
    delete_file(test_url)
    return {"result": top_label}


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch
    model, returns an Numpy array
    """
    # Open the image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))

    return img


def classify_image(image_path):
    # Numpy -> Tensor
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    prob = torch.exp(model.forward(model_input))

    # Top probabilities
    top_p, top_labs = prob.topk(3)
    top_p = top_p.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    print(top_p)
    print(top_labs)
    return top_labs[0]


app.run(host='0.0.0.0', port=5005)
