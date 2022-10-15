from base64 import encodebytes
import io
import json
import mysql.connector
from flask import (
    Blueprint, request, jsonify
)
from PIL import Image
import numpy as np

from . import ranker_helper

bp = Blueprint('s2search', __name__, url_prefix='/s2search')


@bp.route('/', methods=['POST'])
def upload():
    if request.method == 'POST':
        body = request.get_json()
        query = body['query']
        paper_list = body['list']
        print(query)

        scores_w = ranker_helper.get_scores(
            query, paper_list, task_name="Worker", force_global=False)

        print(scores_w)
    return ""
