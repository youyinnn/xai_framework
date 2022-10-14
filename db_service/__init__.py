import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    from . import db_imgnet1000, db_arxiv_cs
    app.register_blueprint(db_imgnet1000.bp)
    app.register_blueprint(db_arxiv_cs.bp)

    return app
