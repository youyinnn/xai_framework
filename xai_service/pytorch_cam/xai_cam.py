import multiprocessing
import asyncio
import os
import base64
import time
import threading
from base64 import encodebytes
import io
import json
from flask import (
    Blueprint, request, jsonify, send_file
)
import numpy as np
import requests
import shutil

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

process_holder = {}

basedir = os.path.abspath(os.path.dirname(__file__))
tmpdir = os.path.join(basedir, 'tmp')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


def cam_task(form_data, task_name):

    # print(form_data)
    # print(task_name)

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

    preprocessing = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    print("# cam gen")
    i = 0

    # explanation save dir
    e_save_dir = os.path.join(tmpdir, form_data['model_name'], form_data['method_name'],
                              form_data['data_set_name'],  form_data['data_set_group_name'],  task_name)

    if not os.path.isdir(e_save_dir):
        os.makedirs(e_save_dir, exist_ok=True)

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
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=None,
                            aug_smooth=True,
                            eigen_smooth=False)[0]

        np.save(os.path.join(e_save_dir, f'{imgd[1]}.npy'), grayscale_cam)

    shutil.make_archive(os.path.join(tmpdir, task_name), 'zip', e_save_dir)
    shutil.rmtree(e_save_dir)


def bytes_to_pil_image(b):
    return Image.open(io.BytesIO(base64.b64decode(b))).convert(
        'RGB')


@bp.route('/', methods=['POST', 'GET'])
def cam_func():
    if request.method == 'GET':
        task_name = request.args['task_name']
        task_time, model_name, method_name, data_set_name, data_set_group_name = task_name.split(
            '|')
        print(task_time, model_name, method_name,
              data_set_name, data_set_group_name)

        return send_file(os.path.join(tmpdir, f'{task_name}.zip'), as_attachment=True)
    if request.method == "POST":
        form_data = request.form
        task_name = f"{time.time()}|{form_data['model_name'].lower()}|{form_data['method_name'].lower()}|{form_data['data_set_name'].lower()}|{form_data['data_set_group_name'].lower()}"
        process = multiprocessing.Process(
            target=cam_task, args=(form_data, task_name))
        process.start()
        process_holder[task_name] = {
            'start_time': time.time(),
            'process': process
        }
        return jsonify({
            'task_name': task_name
        })


def thread_holder_str():
    rs = []
    for tk in process_holder.keys():
        status = 'Running' if process_holder[tk]['process'].is_alive(
        ) else "Stoped"
        # rs.append(f"({tk}, {status})")
        rs.append({
            'task_name': tk,
            'status': status,
            'formated_start_time': time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime(process_holder[tk]['start_time'])),
            'start_time': process_holder[tk]['start_time'],
        })
    return rs


@bp.route('/task', methods=['GET', 'POST'])
def list_task():
    if request.method == 'GET':
        tl = thread_holder_str()
        return jsonify(tl)
    else:
        act = request.args['act']
        if act == 'stop':
            task_name = request.args['task_name']
            process_holder[task_name]['process'].terminate()
            # print(thread_holder_str())
    return ""
