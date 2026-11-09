import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class RequestLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    label = db.Column(db.String(64))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def create_app():
    app = Flask(__name__)
    
    # Config
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET", "dev-secret")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///history.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 # 5 MB limit
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    # Routes
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app