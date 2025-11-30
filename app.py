from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os

# -------------------------------------------------------
# إعداد Flask
# -------------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# -------------------------------------------------------
# تحميل المودل الحقيقي
# -------------------------------------------------------
MODEL_PATH = "risk_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None


# -------------------------------------------------------
# API لحساب المخاطر
# -------------------------------------------------------
@app.route('/api/calculate-risk', methods=['POST'])
def calculate_risk():
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded."
        }), 500

    try:
        data = request.get_json()

        device_type = int(data.get('device_type', 1))
        location_match = int(data.get('location_match', 1))
        time_anomaly = int(data.get('time_anomaly', 0))
        transaction_sensitivity = int(data.get('transaction_sensitivity', 0))
        recent_failed_attempts = int(data.get('recent_failed_attempts', 0))

        features = np.array([[device_type,
                              location_match,
                              time_anomaly,
                              transaction_sensitivity,
                              recent_failed_attempts]])

        risk_score = model.predict(features)[0]
        risk_score = max(0, min(100, int(round(risk_score))))

        if risk_score <= 39:
            level = "low"
            level_ar = "منخفض"
            recommendation = "تنفيذ مباشر - لا توجد مخاطر"
            action = "allow"
        elif risk_score <= 74:
            level = "medium"
            level_ar = "متوسط"
            recommendation = "يتطلب تحقق إضافي (OTP، بصمة)"
            action = "verify"
        else:
            level = "high"
            level_ar = "مرتفع"
            recommendation = "إيقاف العملية ومراجعة أمنية"
            action = "block"

        return jsonify({
            "success": True,
            "risk_score": risk_score,
            "level": level,
            "level_ar": level_ar,
            "recommendation": recommendation,
            "action": action
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# -------------------------------------------------------
# تقديم الواجهة (index.html) من نفس الخدمة
# -------------------------------------------------------
@app.route('/')
def serve_home():
    return send_from_directory('.', 'index.html')


# ملفات الواجهة الأخرى (CSS, JS, images)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


# -------------------------------------------------------
# تشغيل التطبيق
# -------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
