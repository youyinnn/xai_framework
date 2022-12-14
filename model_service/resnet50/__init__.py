import os
import json

from flask import Flask

basedir = os.path.abspath(os.path.dirname(__file__))
tmpdir = os.path.join(basedir, 'tmp')
if not os.path.isdir(tmpdir):
    os.mkdir(tmpdir)


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)

    from . import model
    app.register_blueprint(model.bp)

    return app
