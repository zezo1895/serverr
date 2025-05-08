from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # لقياس الدقة
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# تحميل البيانات وتدريب النموذج عند تشغيل التطبيق
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
print(f"Model Accuracy on Test Data: {accuracy:.4f}")  # طباعة الدقة عند التشغيل

# مسار التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = data['input_data']
        
        input_data_np_array = np.asarray(input_data)
        reshaped_input = input_data_np_array.reshape(1, -1)
        
        prediction = model.predict(reshaped_input)[0]
        
        # إرجاع التنبؤ مع الدقة
        return jsonify({
            'prediction': prediction,
            'accuracy': accuracy  # إرسال الدقة إلى الواجهة الأمامية
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# مسار GET لإرسال رسالة "Hello"
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)

