import torch
import torch.nn as nn
import numpy as np
import librosa
import cv2
import os
import sys
import tempfile
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from torchvision import transforms 
from PIL import Image
from io import BytesIO

# --- IMPORTS FOR DATABASE & AUTH ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Import ffmpeg-python 
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except (ImportError, FileNotFoundError):
    FFMPEG_AVAILABLE = False

# ==============================================================================
# 1. CONFIGURATION & DATABASE SETUP
# ==============================================================================

DEVICE = torch.device("cpu") 

# Cloud Paths (Relative Paths)
FACE_MODEL_PATH = 'best_fer2013_cnn.pth'
SPEECH_MODEL_PATH = 'best_ravdess_speech_cnnlstm (1).pth'
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

app = Flask(__name__)
CORS(app)

# --- DATABASE CONFIGURATION ---
app.config['SECRET_KEY'] = 'your-secret-key-123' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 

# --- USER MODEL ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

# Create DB
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==============================================================================
# 2. MODEL ARCHITECTURES
# ==============================================================================

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.Dropout(0.3),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        return self.classifier(x)

class SpeechEmotionCNN_LSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=8):
        super(SpeechEmotionCNN_LSTM, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, 64, 3, 1), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Conv1d(64, 64, 3, 1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.fc(self.dropout(torch.cat((x[:, -1, :128], x[:, 0, 128:]), dim=1)))

# Initialize Models
try:
    FACE_MODEL = EmotionCNN()
    FACE_MODEL.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=DEVICE))
    FACE_MODEL.eval()
    
    SPEECH_MODEL = SpeechEmotionCNN_LSTM()
    SPEECH_MODEL.load_state_dict(torch.load(SPEECH_MODEL_PATH, map_location=DEVICE))
    SPEECH_MODEL.eval()
    
    FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    print("✅ AI Models Loaded.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")

# Constants
SAMPLE_RATE, N_MFCC, MAX_PAD_LENGTH = 22050, 40, 174
EMOTION_MAP_FACE = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
EMOTION_MAP_SPEECH = {0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'}
FUSION_MAP = {'Neutral': 'Neutral', 'Calm': 'Neutral', 'Happy': 'Happy', 'Sad': 'Sad', 'Angry': 'Angry', 'Fearful': 'Fear', 'Disgust': 'Disgust', 'Surprised': 'Surprise'}

# Helpers
def predict_face_emotion(frame):
    if FACE_CASCADE.empty(): return 'No Face', 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    if len(faces) == 0: return 'No Face', 0.0
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    img = Image.fromarray(cv2.resize(gray[y:y+h, x:x+w], (48,48))).convert('L')
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    with torch.no_grad():
        out = FACE_MODEL(t(img).unsqueeze(0).to(DEVICE))
        prob = torch.softmax(out, 1)
    return EMOTION_MAP_FACE[torch.argmax(prob).item()], torch.max(prob).item()

def predict_speech_emotion(path):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        if len(y) < sr*0.5: return "Silent", 0.0
        mfcc = StandardScaler().fit_transform(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)).T
        if mfcc.shape[0] > MAX_PAD_LENGTH: mfcc = mfcc[:MAX_PAD_LENGTH]
        else: mfcc = np.pad(mfcc, ((0, MAX_PAD_LENGTH-mfcc.shape[0]), (0,0)))
        with torch.no_grad():
            prob = torch.softmax(SPEECH_MODEL(torch.tensor(mfcc).float().unsqueeze(0).to(DEVICE)), 1)
        return EMOTION_MAP_SPEECH[torch.argmax(prob).item()], torch.max(prob).item()
    except: return "Error", 0.0

# ==============================================================================
# 3. ROUTES
# ==============================================================================

@app.route('/')
def serve_html():
    try:
        # Read files
        with open('index.html', 'r', encoding='utf-8') as f: html = f.read()
        
        # Embed JS
        try:
            with open('script.js', 'r', encoding='utf-8') as f: js = f.read()
            html = html.replace('<script src="script.js"></script>', f'<script>{js}</script>')
        except: pass

        # Embed CSS
        try:
            with open('style.css', 'r', encoding='utf-8') as f: css = f.read()
            if '<link rel="stylesheet" href="style.css">' in html:
                html = html.replace('<link rel="stylesheet" href="style.css">', f'<style>{css}</style>')
            else:
                html = html.replace('</head>', f'<style>{css}</style></head>')
        except: pass
        
        return render_template_string(html)
    except: return "Index file missing", 500

# Auth Routes
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    user = User.query.filter_by(email=data.get('email')).first()
    if user: return jsonify({'status': 'error', 'message': 'Email exists'}), 400
    new_user = User(email=data.get('email'), name=data.get('name'), password=generate_password_hash(data.get('password'), method='scrypt'))
    db.session.add(new_user)
    db.session.commit()
    login_user(new_user)
    return jsonify({'status': 'success'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data.get('email')).first()
    if not user or not check_password_hash(user.password, data.get('password')):
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
    login_user(user, remember=True)
    return jsonify({'status': 'success', 'name': user.name})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'status': 'success'})

@app.route('/check_auth')
def check_auth():
    if current_user.is_authenticated:
        return jsonify({'is_logged_in': True, 'name': current_user.name})
    return jsonify({'is_logged_in': False})

# Prediction Route (With Cloud Fix)
@app.route('/predict_video', methods=['POST'])
@login_required
def predict_video():
    # 1. File existence check
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 2. Save file FIRST, then check size (Fixes Cloud 0-byte issue)
    with tempfile.TemporaryDirectory() as temp_dir:
        v_path = os.path.join(temp_dir, file.filename)
        file.save(v_path)
        
        if os.path.getsize(v_path) == 0:
            return jsonify({"error": "File uploaded but size is 0 bytes. Try re-uploading."}), 400

        a_path = os.path.join(temp_dir, 'a.wav')
        
        has_audio = False
        if FFMPEG_AVAILABLE:
            try:
                import ffmpeg
                ffmpeg.input(v_path).output(a_path, format='wav', acodec='pcm_s16le', ar=SAMPLE_RATE, ac=1).overwrite_output().run(quiet=True)
                has_audio = True
            except: pass

        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = int(fps * 2)
        results = []
        cnt = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            cnt += 1
            if cnt % interval == 0:
                f_emo, f_conf = predict_face_emotion(frame)
                s_emo, s_conf = predict_speech_emotion(a_path) if has_audio else ('No Audio', 0.0)
                
                if f_emo == 'No Face': fin_emo, fin_conf = FUSION_MAP.get(s_emo, 'Neutral'), s_conf
                elif f_conf >= s_conf: fin_emo, fin_conf = f_emo, f_conf
                else: fin_emo, fin_conf = FUSION_MAP.get(s_emo, 'Neutral'), s_conf
                
                results.append({"time": round(cnt/fps, 1), "face": f_emo, "voice": s_emo, "fused": fin_emo, "conf": round(fin_conf, 2)})
        cap.release()
        return jsonify({"status": "success", "analysis": results})

if __name__ == '__main__':
    # Cloud Config: 0.0.0.0 and Port 7860
    app.run(debug=False, host='0.0.0.0', port=7860)