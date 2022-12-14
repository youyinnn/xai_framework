from base64 import encodebytes
import io
import os
import base64
import json
import mysql.connector
from flask import (
    Blueprint, request, jsonify, send_file
)
from PIL import Image
import numpy as np
import requests

from pytorch_grad_cam.utils.image import show_cam_on_image

from db_service.tb_image_net_1000 import cnx, get_response_image

bp = Blueprint('explanation', __name__, url_prefix='/db/explanation')

cnx = mysql.connector.connect(
    host="database-1.c0gj2xdlz1ck.us-east-2.rds.amazonaws.com",
    user="xai",
    password="xaidb.2022",
    database="xaifw"
)

basedir = os.path.abspath(os.path.dirname(__file__))


def insert_exp(form):
    cursor = cnx.cursor()
    add_img = (
        "INSERT INTO explanation(model_name, method_name, data_set_name, data_set_group_name, task_name, explanation) VALUES \
            (%(model_name)s,%(method_name)s,%(data_set_name)s, %(data_set_group_name)s, %(task_name)s, %(explanation)s)")
    cursor.execute(add_img, form)
    cnx.commit()
    cursor.close()


def query_exp(form):
    rs = []
    cursor = cnx.cursor()
    add_img = (
        "SELECT * FROM explanation WHERE \
            model_name = %(model_name)s and \
            method_name = %(method_name)s and \
            data_set_name = %(data_set_name)s and \
            data_set_group_name = %(data_set_group_name)s and \
            task_name = %(task_name)s")
    cursor.execute(add_img, form)
    for row in cursor:
        rs.append(row)
    cursor.close()
    return rs


@bp.route('/', methods=['POST'])
def add_explanation():
    if request.method == 'POST':
        form = dict(request.form)
        form['explanation'] = request.files.get('explanation').read()
        insert_exp(form)
    return ""


def bytes_to_pil_image(b):
    return Image.open(io.BytesIO(base64.b64decode(b)))


@bp.route('/', methods=['GET'])
def get_explanation():
    if request.method == 'GET':
        form = request.args
        rs = query_exp(form)
        exp = rs[0][-1]
        exp_output_path = os.path.join(basedir, 'tmp', 'explanation.npz')
        f = open(exp_output_path, 'wb')
        f.write(exp)
        # exp = np.load(exp_output_path)

        # grayscale_cam = exp['arr_0']

        # cursor = cnx.cursor()
        # l = []
        # q = (
        #     f"SELECT id, img_name, img_data, img_group, img_label FROM image_net_1000 WHERE img_group = 't2'")
        # # print(q)
        # cursor.execute(q)
        # for (id, img_name, img_data, img_group, img_label) in cursor:
        #     l.append((id, img_name, get_response_image(
        #         img_data), img_group, img_label))
        #     # Image.open(io.BytesIO(img_data)).show()

        # cursor.close()

        # rgb_img = bytes_to_pil_image(l[0][2])

        # cam_image = show_cam_on_image(np.float32(
        #     rgb_img) / 255, grayscale_cam[0], use_rgb=True)
        # Image.fromarray(cam_image).show()
    return send_file(exp_output_path, as_attachment=True)
