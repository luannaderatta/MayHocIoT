from flask import Flask, render_template, request, url_for, jsonify
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
import joblib
import json
import paho.mqtt.client as mqtt # <--- THÊM THƯ VIỆN MQTT

app = Flask(__name__)

# --- CẤU HÌNH FLASK & DB ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# ---------------------------------------------------------
# [PHẦN MỚI] CẤU HÌNH MQTT & IOT STATE
# ---------------------------------------------------------

# Cấu hình Broker (Dùng public broker miễn phí)
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
# Topic này phải TRÙNG với topic trong code ESP32 trên Wokwi
MQTT_TOPIC_SUB = "smart-fridge/sensor/data" 

# Biến toàn cục lưu trạng thái nhiệt độ (Để hiển thị lên web)
fridge_state = {
    'freezer': -18,
    'cooler': 4,
    'vegetable': 8
}

# 1. Hàm khi kết nối thành công
def on_connect(client, userdata, flags, rc):
    print(f"🔌 [MQTT] Đã kết nối Broker! Mã: {rc}")
    # Đăng ký nhận tin nhắn từ topic cảm biến
    client.subscribe(MQTT_TOPIC_SUB)

# 2. Hàm khi nhận được tin nhắn từ ESP32
def on_message(client, userdata, msg):
    global fridge_state
    try:
        payload = msg.payload.decode()
        print(f"📩 [MQTT] Nhận: {payload}")
        
        # Giải mã JSON từ Wokwi (VD: {"zone": "cooler", "temp": 5})
        data = json.loads(payload)
        zone = data.get('zone')
        temp = data.get('temp')
        
        # Cập nhật vào bộ nhớ
        if zone in fridge_state:
            fridge_state[zone] = int(temp)
            
    except Exception as e:
        print(f"❌ [MQTT] Lỗi dữ liệu: {e}")

# 3. Hàm khởi động MQTT chạy ngầm
def start_mqtt():
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        
        print("⏳ [MQTT] Đang kết nối...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # loop_start() giúp MQTT chạy trên luồng riêng, không chặn Flask
        client.loop_start() 
    except Exception as e:
        print(f"❌ Không thể chạy MQTT: {e}")

# Kích hoạt MQTT ngay
start_mqtt()

# ---------------------------------------------------------
# KẾT THÚC PHẦN MQTT
# ---------------------------------------------------------

# --- DỮ LIỆU HẠN SỬ DỤNG ---
SHELF_LIFE_DB = {
    'apple': 14, 'banana': 5, 'beef': 3, 'bell pepper': 10, 
    'broccoli': 5, 'cabbage': 14, 'carrot': 21, 'cauliflower': 7, 
    'chicken': 3, 'cucumber': 7, 'egg': 30, 'fish': 2, 
    'mango': 7, 'orange': 21, 'potato': 60, 'tomato': 7
}

# --- LOAD MODELS ---
print("⏳ Đang khởi động hệ thống AI...")
model = None
rf_model = None
try:
    model_path = os.path.join(BASE_DIR, 'best_my_fridge_model.pt')
    if os.path.exists(model_path):
        model = YOLO(model_path)
    
    rf_path = os.path.join(BASE_DIR, 'random_forest_model.pkl')
    enc_path = os.path.join(BASE_DIR, 'ingredients_encoder.pkl')
    json_path = os.path.join(BASE_DIR, 'recipes_info.json')

    if os.path.exists(rf_path) and os.path.exists(enc_path) and os.path.exists(json_path):
        rf_model = joblib.load(rf_path)
        ingredients_encoder = joblib.load(enc_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)
    print("✅ Hệ thống AI đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi tải Model: {e}")

# --- DATABASE MODEL ---
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100))
    original_filename = db.Column(db.String(100))
    items = db.Column(db.String(500))
    suggested_dish = db.Column(db.String(200))
    date_posted = db.Column(db.DateTime, default=datetime.now)

with app.app_context():
    db.create_all()

# --- HELPER FUNCTIONS ---
def normalize_text(text):
    return text.lower().strip() if text else ""

def get_recipe_suggestions(detected_items):
    if not rf_model or not detected_items: return []
    suggestions = []
    current_ings_clean = set([normalize_text(i) for i in detected_items])
    try:
        input_vector = ingredients_encoder.transform([list(current_ings_clean)])
        probs = rf_model.predict_proba(input_vector)[0]
        top_indices = np.argsort(probs)[-5:][::-1]
        for idx in top_indices:
            if probs[idx] > 0.05:
                recipe = recipes_data[idx]
                recipe_ings = set(recipe['ingredients'])
                missing = [ing for ing in recipe_ings if ing not in current_ings_clean]
                match_score = int((1 - len(missing)/len(recipe_ings)) * 100)
                suggestions.append({
                    'name': recipe['dish_name_vn'],
                    'time': recipe['cooking_time'],
                    'difficulty': recipe['difficulty'],
                    'missing': missing,
                    'match': match_score,
                    'steps': recipe['cooking_steps']
                })
        suggestions.sort(key=lambda x: x['match'], reverse=True)
    except Exception: pass
    return suggestions

def check_expiry(items):
    expiry_list = []
    now = datetime.now()
    for item in items:
        days = SHELF_LIFE_DB.get(normalize_text(item), 7)
        exp_date = now + timedelta(days=days)
        status = "success"
        if days <= 2: status = "danger"
        elif days <= 5: status = "warning"
        expiry_list.append({'name': item, 'days_left': days, 'date': exp_date.strftime("%d/%m"), 'status': status})
    return expiry_list

# --- ROUTES ---

# [API MỚI] Để giao diện Web lấy nhiệt độ cập nhật từ MQTT
@app.route('/get_temp_state')
def get_temp_state():
    return jsonify(fridge_state)

# [API MỚI] Để giao diện Web cập nhật nhiệt độ (khi kéo thanh trượt)
@app.route('/update_temp', methods=['POST'])
def update_temp():
    data = request.json
    zone = data.get('zone')
    temp = data.get('temp')
    
    if zone in fridge_state:
        fridge_state[zone] = int(temp)
        # Nếu muốn web điều khiển ngược lại thiết bị, 
        # bạn có thể thêm lệnh client.publish(...) ở đây
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 400

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {'uploaded_img': None, 'result_img': None, 'items': [], 'expiry': [], 'suggestions': [], 'error': None}

    if request.method == 'POST':
        if 'file' not in request.files:
            context['error'] = "Chưa chọn file!"
        else:
            file = request.files['file']
            if file.filename != '':
                try:
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    filename = f"upload_{timestamp}_{file.filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    context['uploaded_img'] = url_for('static', filename=f'uploads/{filename}')

                    if model:
                        results = model(filepath, conf=0.5)
                        res_plotted = results[0].plot()
                        res_filename = f"pred_{filename}"
                        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], res_filename), res_plotted)
                        context['result_img'] = url_for('static', filename=f'uploads/{res_filename}')
                        
                        items = list(set([model.names[int(box.cls[0])] for box in results[0].boxes]))
                        context['items'] = items
                        context['expiry'] = check_expiry(items)
                        context['suggestions'] = get_recipe_suggestions(items)
                        
                        top_dish = context['suggestions'][0]['name'] if context['suggestions'] else "Không có"
                        new_record = History(
                            filename=res_filename,
                            original_filename=filename,
                            items=", ".join(items),
                            suggested_dish=top_dish
                        )
                        db.session.add(new_record)
                        db.session.commit()
                    else:
                        context['error'] = "Model chưa được load!"
                except Exception as e:
                    context['error'] = f"Lỗi xử lý: {str(e)}"

    return render_template('index.html', **context)

@app.route('/history')
def history():
    try:
        records = History.query.order_by(History.date_posted.desc()).all()
    except Exception:
        records = []
    return render_template('history.html', records=records)

if __name__ == '__main__':
    app.run(debug=True, port=5000)