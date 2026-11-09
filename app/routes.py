from flask import Blueprint, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
from . import db, RequestLog
from .inference import predict

main_bp = Blueprint("main", __name__)
ALLOWED = {"image/jpeg", "image/png", "image/webp"}

@main_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("image")
        
        if not f or f.filename == "":
            return render_template("error.html", msg="No file selected."), 400
            
        if f.mimetype not in ALLOWED:
            return render_template("error.html", msg="Unsupported file type. Upload a JPG, PNG, or WebP."), 400
            
        raw = f.read()
        
        try:
            Image.open(BytesIO(raw)).verify()
        except Exception:
            return render_template("error.html", msg="Corrupt or invalid image file."), 400
            
        label, conf, prob = predict(raw)
        safe_name = secure_filename(f.filename)
        
        # Save to Database (Task 5)
        log = RequestLog(filename=safe_name, label=label, confidence=conf)
        db.session.add(log)
        db.session.commit()
        
        return render_template("result.html", label=label, conf=round(conf*100, 2), filename=safe_name, raw_p=round(prob*100, 2))
        
    return render_template("upload.html")

@main_bp.route("/history")
def history():
    logs = RequestLog.query.order_by(RequestLog.id.desc()).limit(50).all()
    return render_template("history.html", logs=logs)

@main_bp.route("/health")
def health():
    return jsonify(status="ok", model="visionguard-v1", threshold=0.15)