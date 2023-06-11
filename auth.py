from flask import Blueprint, current_app
from flask import current_app as app
from flask_sqlalchemy import SQLAlchemy

auth_bp = Blueprint('auth', __name__, template_folder='templates')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    print(app.name)
    db = SQLAlchemy()

    return 'login'