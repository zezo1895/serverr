from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # تحديد CORS لمسار /predict

# إعداد التسجيل (Logging)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# التحقق من وجود ملف البيانات
if not os.path.exists('Copy of sonar data (1).csv'):
    raise FileNotFoundError("ملف البيانات 'Copy of sonar data (1).csv' غير موجود!")

# تحميل البيانات وتدريب النموذج
solar_data = pd.read_csv('Copy of sonar data (1).csv', header=None)
X = solar_data.drop(columns=60, axis=1)
y = solar_data[60]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)

# تدريب النموذج
model = LogisticRegression()
model.fit(X_train, y_train)

# حساب دقة النموذج على بيانات الاختبار
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Model Accuracy on Test Data: {accuracy:.4f}")

# مسار التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'input_data' not in data:
            return jsonify({'error': 'مطلوب حقل input_data في البيانات المرسلة'}), 400
        
        input_data = data['input_data']
        if len(input_data) != 60:
            return jsonify({'error': 'البيانات المدخلة يجب أن تحتوي على 60 قيمة'}), 400
        
        input_data_np_array = np.asarray(input_data)
        reshaped_input = input_data_np_array.reshape(1, -1)
        
        prediction = model.predict(reshaped_input)[0]
        
        logger.info(f"Prediction made: {prediction}")
        return jsonify({
            'prediction': prediction,
            'accuracy': accuracy
        })
    except Exception as e:
        logger.error(f"خطأ أثناء التنبؤ: {str(e)}")
        return jsonify({'error': str(e)}), 500

# مسار GET لإرسال رسالة "Hello"
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

# مسار لعرض معلومات النموذج
@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model': 'Logistic Regression',
        'test_accuracy': accuracy,
        'data_shape': X_train.shape
    })

if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))  # استخدام PORT من البيئة أو 5000 افتراضيًا
    serve(app, host='0.0.0.0', port=port)