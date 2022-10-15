import io
import os
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import (
    Blueprint, request, jsonify
)


# Download this file <https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json>_ as imagenet_class_index.json
#imagenet_class_index = json.load(open('./static/imagenet_class_index.json'))

# model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# model = models.resnet50(pretrained=True)
model.eval()
#target_layers = model.layer4[-1]

# preprocess for transform
preprocess = transforms.Compose([
    transforms.Resize([256, ]),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

bp = Blueprint('resnet50', __name__, url_prefix='/resnet50')

basedir = os.path.abspath(os.path.dirname(__file__))

print(basedir)


@bp.route('/', methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        # Read the image in PIL format

        image = request.files["image"].read()
        img = Image.open(io.BytesIO(image))

        # Preprocess the image and prepare it for classification.

        batch = preprocess(img).unsqueeze(0)
        prediction = model(batch).squeeze(0).softmax(0)

        # Use the model and print the predicted category
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        with open(os.path.join(basedir, "static", "imagenet_classes.txt"), "rb") as f:
            categories = [s.strip() for s in f.readlines()]
            category_name = categories[class_id]
        print(f"{category_name}: {100 * score}%")
        return jsonify({'category': category_name, 'score': score})

    elif request.method == 'GET':
        return model
