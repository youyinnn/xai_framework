import asyncio
import os
import base64
import time
import threading
from base64 import encodebytes
import io
import json
from flask import (
    Blueprint, request, jsonify
)
import numpy as np
import requests

from torchvision import models
import torch.nn as nn
import torch.cuda as cuda
import torch
import torchvision.transforms as T
import cv2
from PIL import Image

from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise


from pytorch_grad_cam.utils.image import show_cam_on_image

bp = Blueprint('pt_cam', __name__, url_prefix='/xai/pt_cam')

thread_holder = {}

basedir = os.path.abspath(os.path.dirname(__file__))
tmpdir = os.path.join(basedir, 'tmp')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


class MyThread(threading.Thread):
    def __init__(self, run_func, run_func_args, *args, **kwargs):
        super(MyThread, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        self.run_func = run_func
        self.run_func_args = run_func_args

    # function using _stop function
    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        if self.stopped():
            return
        self.run_func(**self.run_func_args)


def cam_task(form_data, task_name):
    print('# get image data')
    response = requests.get(
        form_data['db_service_url'], params={
            'img_group': form_data['data_set_group_name'],
            'with_img_data': 1,
        })
    # print(response)
    img_data = json.loads(response.content.decode('utf-8'))

    # for igd in img_data:
    #     dcode = base64.b64decode(igd[2])
    #     img = Image.open(io.BytesIO(dcode))
    #     img.show()

    print('# get model pt')
    model_pt_path = os.path.join(tmpdir, f"{form_data['model_name']}.pth")
    response = requests.get(
        form_data['model_service_url'])
    with open(model_pt_path, "wb") as f:
        f.write(response.content)

    # load model

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    model.load_state_dict(torch.load(model_pt_path))

    target_layers = [model.layer4]

    def get_img_np(path):
        d = cv2.imread(path, 1)[:, :, ::-1]
        return np.float32(d) / 255

    preprocessing = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    grayscale_cam = []

    print("# cam gen")
    i = 0
    for imgd in img_data:
        print(i, imgd[1])
        i += 1
        rgb_img = bytes_to_pil_image(imgd[2])

        input_tensor = torch.tensor(np.array([
            preprocessing(x).numpy()
            for x in [rgb_img]
        ]))

        cam = GradCAM(model=model,
                      target_layers=target_layers,
                      use_cuda=torch.cuda.is_available())

        cam.batch_size = 32

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        grayscale_cam = [imgd[1], cam(input_tensor=input_tensor,
                                      targets=None,
                                      aug_smooth=True,
                                      eigen_smooth=False)[0]]

        exp_output_path = os.path.join(tmpdir, 'exp.npz')

        np.savez_compressed(exp_output_path, grayscale_cam)

        payload = {'model_name': form_data['model_name'],
                   'method_name': form_data['method_name'],
                   'data_set_name': form_data['data_set_name'],
                   'data_set_group_name': form_data['data_set_group_name'],
                   'task_name': task_name}
        files = [
            ('explanation', ('exp.npz',
                             open(exp_output_path, 'rb'), 'application/octet-stream'))
        ]
        headers = {}

        response = requests.request(
            "POST", form_data['explanation_db_service_url'], headers=headers, data=payload, files=files)

        print(i, response.text)

    thread_holder[task_name].stop()


def bytes_to_pil_image(b):
    return Image.open(io.BytesIO(base64.b64decode(b))).convert(
        'RGB')


@bp.route('/', methods=['POST'])
def upload_paper():
    if request.method == "POST":
        form_data = request.form
        task_name = f"{time.time()}-{form_data['model_name'].lower()}-{form_data['method_name'].lower()}-{form_data['data_set_name'].lower()}-{form_data['data_set_group_name'].lower()}"
        t = MyThread(cam_task, {
            'task_name': task_name,
            'form_data': form_data
        })
        t.start()
        thread_holder[task_name] = t
    return "done"


def thread_holder_str():
    rs = []
    for tk in thread_holder.keys():
        status = 'Running' if not thread_holder[tk].stopped() else "Stoped"
        # rs.append(f"({tk}, {status})")
        rs.append({
            'name': tk,
            'status': status
        })
    return rs


@bp.route('/task', methods=['GET', 'POST'])
def list_task():
    if request.method == 'GET':
        print(thread_holder_str())
        for tk in thread_holder.keys():
            t = thread_holder[tk]
    else:
        act = request.args['act']
        if act == 'stop':
            thread_name = request.args['name']
            thread_holder[thread_name].stop()
            print(thread_holder_str())
    return ""
