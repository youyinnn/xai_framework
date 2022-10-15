import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

import requests

from torchvision import models

#function to transform image
# def transform_image(image_bytes):
#     my_transforms = transforms.Compose([transforms.Resize(255),
#                                         transforms.CenterCrop(224),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(
#                                             [0.485, 0.456, 0.406],
#                                             [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)


# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])


#Download this file <https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json>_ as imagenet_class_index.json
imagenet_class_index = json.load(open('./static/imagenet_class_index.json'))

#function to get the prediction
#model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = models.resnet50(pretrained=True)
#target_layers = model.layer4[-1]

# def get_prediction(image_bytes):
#     tensor = transform_image(image_bytes=image_bytes)
#     outputs = model.forward(tensor)
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return imagenet_class_index[predicted_idx]

# with open("./static/img/dog.jpeg", 'rb') as f:
#     image_bytes = f.read()
#     print(get_prediction(image_bytes=image_bytes))


app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/model', methods=['GET','POST'])
def modelto():
    if request.method == 'POST':
        # Read the image in PIL format
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        # Preprocess the image and prepare it for classification.
        image = prepare_image(image, target_size=(224, 224))

        # Classify the input image and then initialize the list of predictions to return to the client.
        preds = F.softmax(model(image), dim=1)
        results = torch.topk(preds.cpu().data, k=3, dim=1)

        data['predictions'] = list()

        # Loop over the results and add them to the list of returned predictions
        for prob, label in zip(results[0][0], results[1][0]):
            label_name = idx2label[label]
            r = {"label": label_name, "probability": float(prob)}
            data['predictions'].append(r)

        # Indicate that the request was a success.
        data["success"] = True

        return flask.jsonify(data)

    elif request.method == 'get':
        return model
