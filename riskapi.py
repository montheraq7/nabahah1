import numpy as np
from sklearn.ensemble import RandomForestRegressor

# تركيب بيانات تدريبية واقعية - 5000 نقطة
np.random.seed(42)
n_samples = 5000

# توليد البيانات
device_type = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
location_match = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])
time_anomaly = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
transaction_sensitivity = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])
recent_failed_attempts = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, 
                                          p=[0.5, 0.2, 0.15, 0.1, 0.04, 0.01])

X_train = np.column_stack([device_type, location_match, time_anomaly, 
                           transaction_sensitivity, recent_failed_attempts])

# حساب Risk Score
risk_scores = []
for i in range(n_samples):
    base_score = 10
    base_score += recent_failed_attempts[i] * 14
    base_score += transaction_sensitivity[i] * 12.5
    if time_anomaly[i] == 1:
        base_score += 20
    if location_match[i] == 0:
        base_score += 10
    if device_type[i] == 0:
        base_score += 10
    noise = np.random.normal(0, 3)
    base_score += noise
    risk_scores.append(max(0, min(100, base_score)))

y_train = np.array(risk_scores)

# تدريب المودل
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# طباعة أهمية الميزات
feature_names = ['device_type', 'location_match', 'time_anomaly', 
                'transaction_sensitivity', 'recent_failed_attempts']
print("\n" + "="*70)
print("📊 FEATURE IMPORTANCE ANALYSIS")
print("="*70)
for name, importance in zip(feature_names, model.feature_importances_):
    bar = "█" * int(importance * 50)
    print(f"{name:28} {importance:.4f} ({importance*100:5.2f}%) {bar}")

print(f"\n🎯 Model R² Score: {model.score(X_train, y_train):.4f}")

# اختبار سيناريوهات مختلفة
print("\n" + "="*70)
print("🧪 TEST SCENARIOS")
print("="*70)

test_cases = [
    {
        'name': '✅ Low Risk - Known device, matching location',
        'features': [1, 1, 0, 0, 0]  # جهاز معروف، موقع مطابق، وقت عادي، حساسية منخفضة، لا محاولات فاشلة
    },
    {
        'name': '⚠️  Medium Risk - Unknown device',
        'features': [0, 1, 0, 1, 0]  # جهاز غير معروف، موقع مطابق، حساسية متوسطة
    },
    {
        'name': '⚠️  Medium Risk - Different location',
        'features': [1, 0, 0, 1, 0]  # جهاز معروف، موقع مختلف، حساسية متوسطة
    },
    {
        'name': '⚠️  Medium Risk - Unusual time + sensitive',
        'features': [1, 1, 1, 2, 0]  # وقت غير عادي، عملية حساسة
    },
    {
        'name': '🚨 High Risk - Failed attempts',
        'features': [1, 1, 0, 1, 3]  # 3 محاولات فاشلة
    },
    {
        'name': '🚨 High Risk - Multiple risk factors',
        'features': [0, 0, 1, 2, 2]  # جهاز غير معروف، موقع مختلف، وقت شاذ، حساسية عالية، محاولات فاشلة
    },
    {
        'name': '🚨 Very High Risk - Many failed attempts',
        'features': [0, 0, 1, 2, 5]  # 5 محاولات فاشلة + عوامل أخرى
    },
]

for test in test_cases:
    features = np.array([test['features']])
    risk_score = int(round(model.predict(features)[0]))
    risk_score = max(0, min(100, risk_score))
    
    if risk_score <= 39:
        level = "LOW"
        emoji = "🟢"
    elif risk_score <= 74:
        level = "MEDIUM"
        emoji = "🟡"
    else:
        level = "HIGH"
        emoji = "🔴"
    
    print(f"\n{test['name']}")
    print(f"   Features: Device={test['features'][0]}, Location={test['features'][1]}, "
          f"Time={test['features'][2]}, Sensitivity={test['features'][3]}, FailedAttempts={test['features'][4]}")
    print(f"   {emoji} Risk Score: {risk_score}/100 ({level})")

print("\n" + "="*70)
print("📈 DATA DISTRIBUTION")
print("="*70)
print(f"Total samples: {len(y_train)}")
print(f"Risk Score range: {y_train.min():.1f} - {y_train.max():.1f}")
print(f"Risk Score mean: {y_train.mean():.1f}")
print(f"Risk Score std: {y_train.std():.1f}")

low_risk = np.sum(y_train <= 39)
medium_risk = np.sum((y_train > 39) & (y_train <= 74))
high_risk = np.sum(y_train > 74)

print(f"\n🟢 Low Risk (0-39):    {low_risk:4d} samples ({low_risk/len(y_train)*100:.1f}%)")
print(f"🟡 Medium Risk (40-74): {medium_risk:4d} samples ({medium_risk/len(y_train)*100:.1f}%)")
print(f"🔴 High Risk (75-100):  {high_risk:4d} samples ({high_risk/len(y_train)*100:.1f}%)")
print("="*70)
