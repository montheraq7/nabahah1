from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder='.')
CORS(app)

# -------------------------------------------------------
# تحميل المودل الحقيقي من risk_model.pkl
# -------------------------------------------------------
MODEL_PATH = "risk_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Loaded model successfully:", MODEL_PATH)
except Exception as e:
    print("❌ Failed to load model:", str(e))
    model = None


@app.route('/api/calculate-risk', methods=['POST'])
def calculate_risk():
    """
    API لحساب مؤشر المخاطر
    """

    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Check risk_model.pkl"
        }), 500

    try:
        data = request.get_json()

        # استخراج القيم
        device_type = int(data.get('device_type', 1))
        location_match = int(data.get('location_match', 1))
        time_anomaly = int(data.get('time_anomaly', 0))
        transaction_sensitivity = int(data.get('transaction_sensitivity', 0))
        recent_failed_attempts = int(data.get('recent_failed_attempts', 0))

        # تجهيز الـ vector
        features = np.array([[device_type,
                              location_match,
                              time_anomaly,
                              transaction_sensitivity,
                              recent_failed_attempts]])

        # التنبؤ
        risk_score = model.predict(features)[0]

        # قيد 0-100
        risk_score = max(0, min(100, int(round(risk_score))))

        # منطق التصنيف
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
            'success': True,
            'risk_score': risk_score,
            'level': level,
            'level_ar': level_ar,
            'recommendation': recommendation,
            'action': action
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "Risk API running"
    })


@app.route('/')
def home():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
