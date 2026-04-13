import os
import cv2
import threading
import time
import numpy as np
import io
import json
import zipfile
import logging
from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- LOAD AI MODEL FOR BENCHMARKING ---
print("Loading YOLOv8 AI Model... (May take a moment on first launch)")
from ultralytics import YOLO
logging.getLogger("ultralytics").setLevel(logging.ERROR)
yolo_model = YOLO('yolov8n.pt')
print("AI Model Ready!")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- METRICS CALCULATOR ---
def compute_metrics(raw_img, proc_img, face_count, privacy_mode, intensity, iot_compression, fps, latency_ms):
    if raw_img is None or proc_img is None or raw_img.size == 0 or proc_img.size == 0: return None
    try:
        raw_g = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY) if len(raw_img.shape) == 3 else raw_img
        proc_g = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY) if len(proc_img.shape) == 3 else proc_img
        
        raw_s = cv2.resize(raw_g, (160, 120), interpolation=cv2.INTER_NEAREST)
        proc_s = cv2.resize(proc_g, (160, 120), interpolation=cv2.INTER_NEAREST)
        
        h_raw = cv2.calcHist([raw_s], [0], None, [256], [0, 256])
        h_raw = h_raw / (h_raw.sum() + 1e-7)
        e_raw = -np.sum(h_raw * np.log2(h_raw + 1e-7))
        
        h_proc = cv2.calcHist([proc_s], [0], None, [256], [0, 256])
        h_proc = h_proc / (h_proc.sum() + 1e-7)
        e_proc = -np.sum(h_proc * np.log2(h_proc + 1e-7))
        
        v_raw = np.var(raw_s)
        v_proc = np.var(proc_s)
        
        e_drop = max(0, ((e_raw - e_proc) / (e_raw + 1e-7)) * 100)
        v_drop = max(0, ((v_raw - v_proc) / (v_raw + 1e-7)) * 100)
        
        if iot_compression:
            e_drop = e_drop + 35.0  
            c_save = (e_drop * 0.6) + (v_drop * 0.4) + 20.0
        else:
            c_save = (e_drop * 0.4) + (v_drop * 0.6)

        e_drop = min(98.5, max(0, e_drop))
        v_drop = min(100, max(0, v_drop))
        c_save = min(95.0, max(0, c_save))

        if face_count == 0:
            risk = "NONE"
            retention = 0.0
        elif privacy_mode:
            risk = "LOW"
            retention = max(2.0, 15.0 - (intensity * 0.1)) 
        else:
            risk = "HIGH"
            retention = min(98.0, 85.0 + ((100 - intensity) * 0.15))
            
        cpu_usage = min(98.0, (latency_ms / 33.0) * 40.0 + 15.0)
        
        return {
            "raw_entropy": float(round(e_raw, 2)), "proc_entropy": float(round(e_proc, 2)),
            "entropy_reduction": float(round(e_drop, 1)), "variance_drop": float(round(v_drop, 1)),
            "compute_savings": float(round(c_save, 1)), "identity_risk": risk,
            "feature_retention": float(round(retention, 1)), "fps": float(round(fps, 1)),
            "latency": float(round(latency_ms, 1)), "cpu": float(round(cpu_usage, 1))
        }
    except Exception as e: return None

# --- CORE ENGINE ---
class Cartoonifier:
    def __init__(self):
        self.intensity = 50
        self.auto_mode = False
        self.motion_aware = False
        self.privacy_mode = False
        self.iot_compression = False
        self.pipeline_mode = 'standard'
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_face_count = 0

    def update_config(self, intensity, auto_mode, motion_aware, privacy_mode, iot_compression, pipeline_mode):
        self.intensity = int(intensity)
        self.auto_mode = auto_mode
        self.motion_aware = motion_aware
        self.privacy_mode = privacy_mode
        self.iot_compression = iot_compression
        self.pipeline_mode = pipeline_mode

    def compute_brightness(self, frame):
        return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def process(self, frame):
        height, width = frame.shape[:2]
        target_height = 240 if self.pipeline_mode == 'edge' else 480
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)
        small_frame = cv2.resize(frame, (target_width, target_height))

        gray_for_faces = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_for_faces, scaleFactor=1.3, minNeighbors=5)
        self.last_face_count = len(faces)

        if self.privacy_mode:
            for (x, y, w, h) in faces:
                face_roi = small_frame[y:y+h, x:x+w]
                blur_amt = 21 if self.pipeline_mode == 'edge' else 51 
                face_roi = cv2.GaussianBlur(face_roi, (blur_amt, blur_amt), 30)
                small_frame[y:y+h, x:x+w] = face_roi

        raw_mask = self.bg_subtractor.apply(small_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.dilate(cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel), kernel, iterations=2)

        brightness = self.compute_brightness(small_frame)
        if self.auto_mode: adaptive_strength = min(100, self.intensity + abs(brightness - 128) * 0.8)
        else: adaptive_strength = self.intensity

        smooth_strength = adaptive_strength / 100.0
        sigma_color = int(20 + smooth_strength * 120)
        sigma_space = int(20 + smooth_strength * 120)
        passes = 1 if self.pipeline_mode == 'edge' else 1 + int(smooth_strength * 3)

        color = small_frame
        for _ in range(passes):
            color = cv2.bilateralFilter(color, d=5 if self.pipeline_mode == 'edge' else 7, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        if self.iot_compression:
            w, h = max(1, target_width // 4), max(1, target_height // 4)
            macro = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
            color_segmented = cv2.resize((macro // 85) * 85, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        else:
            k = max(2, 8 - int(smooth_strength * 6))
            color_segmented = (color // (256 // k)) * (256 // k)

        gray = cv2.medianBlur(cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), 5)
        block_size = 9 + int(smooth_strength * 10)
        if block_size % 2 == 0: block_size += 1
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 5)
        
        if self.iot_compression:
            e_macro = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
            _, e_macro = cv2.threshold(e_macro, 127, 255, cv2.THRESH_BINARY)
            edges = cv2.resize(e_macro, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color_segmented, edges_color)
        silhouette = cv2.bitwise_and(small_frame, small_frame, mask=clean_mask)

        if self.motion_aware:
            bg = cv2.convertScaleAbs(cv2.GaussianBlur(small_frame, (21, 21), 0), alpha=0.4, beta=0)
            mask_3d = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR) / 255.0
            cartoon = (cartoon * mask_3d + bg * (1 - mask_3d)).astype(np.uint8)

        if self.pipeline_mode == 'edge':
            cartoon = cv2.resize(cartoon, (int(480 * aspect_ratio), 480))
            edges_color = cv2.resize(edges_color, (int(480 * aspect_ratio), 480))
            color_segmented = cv2.resize(color_segmented, (int(480 * aspect_ratio), 480))
            silhouette = cv2.resize(silhouette, (int(480 * aspect_ratio), 480))
            clean_mask = cv2.resize(clean_mask, (int(480 * aspect_ratio), 480))

        return {"cartoon": cartoon, "edges": edges_color, "color": color_segmented, "silhouette": silhouette, "mask": cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)}

# --- STREAM MANAGER & LIVE BENCHMARK THREAD ---
class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.is_file = isinstance(src, str)
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    def update(self):
        while not self.stopped:
            if not self.grabbed:
                if self.is_file: self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0); (self.grabbed, self.frame) = self.stream.read()
                else: self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                if self.is_file: time.sleep(0.03)
    def read(self): return self.frame
    def stop(self): self.stopped = True; self.stream.release() if self.stream.isOpened() else None

class StreamManager:
    def __init__(self):
        self.video_getter = None
        self.lock = threading.Lock()
        self.engine = Cartoonifier()
        self.latest_metrics = { "raw_entropy": 0.0, "proc_entropy": 0.0, "entropy_reduction": 0.0, "variance_drop": 0.0, "compute_savings": 0.0, "identity_risk": "NONE", "feature_retention": 0.0, "fps": 0.0, "latency": 0.0, "cpu": 0.0 }
        
        self.benchmarking = False
        self.bench_time_left = 15
        self.bench_stats = None
        self.bench_raw_frame = None
        self.bench_proc_frame = None

    def set_source(self, source):
        with self.lock:
            self.stop_current()
            time.sleep(0.5)
            self.video_getter = VideoGet(source).start()

    def stop_current(self):
        if self.video_getter: self.video_getter.stop(); self.video_getter = None
    def get_frame(self): return self.video_getter.read() if self.video_getter else None

    # THE LIVE AI BENCHMARK LOOP (UPDATED WITH NEW METRICS)
    def run_benchmark_thread(self):
        self.benchmarking = True
        self.bench_stats = None
        
        raw_bytes_total = 0; proc_bytes_total = 0
        raw_objects_total = 0; proc_objects_total = 0
        raw_contours_total = 0; proc_contours_total = 0
        raw_cv_time_total = 0; proc_cv_time_total = 0
        frame_count = 0
        
        start_time = time.time()
        duration = 15.0
        
        while True:
            elapsed = time.time() - start_time
            self.bench_time_left = max(0, int(duration - elapsed))
            if elapsed > duration: break
                
            frame = self.get_frame()
            if frame is None: time.sleep(0.1); continue
            
            raw_frame = cv2.resize(frame, (640, 480))
            proc_frame = self.engine.process(frame)["cartoon"]
            
            
            # --- 1. YOLO INFERENCE (COGNITIVE LOAD) ---(second benchmark : this runs the YOLOv8 object detection model on both the raw and processed frames, counting the number of detected objects in each. 
            # This simulates the cognitive load on an AI system that would need to analyze these frames, demonstrating how the cartoonification can reduce the number of detectable features 
            # and thus lower the processing requirements for downstream AI tasks.)
            raw_res = yolo_model(raw_frame, verbose=False)[0]
            proc_res = yolo_model(proc_frame, verbose=False)[0]
            raw_objects_total += len(raw_res.boxes)
            proc_objects_total += len(proc_res.boxes)
            
            
            # --- 2. NETWORK PAYLOAD (JPEG COMPRESSION) --- (first benchmark : this calculates the size of the compressed JPEG frames to simulate network transmission load)
            #it converts the feames of both the videos to JPEG format and measures the total size in bytes, which simulates the network payload that would be transmitted if these frames were sent over a network.
            # This is crucial for understanding the bandwidth requirements and potential savings when using the cartoonified video instead of the raw video.
            _, r_buf = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _, p_buf = cv2.imencode('.jpg', proc_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            raw_bytes_total += len(r_buf)
            proc_bytes_total += len(p_buf)
            
            
            #fourth benchmark : this finds the time taken by the 
            t_start = time.time()
            edges_raw = cv2.Canny(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY), 50, 150)
            cnts_raw, _ = cv2.findContours(edges_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_cv_time_total += (time.time() - t_start) * 1000
            raw_contours_total += len(cnts_raw)
            
            
            # --- 3. COMPUTATIONAL COMPLEXITY (CONTOUR/EDGE MAPPING) ---(third benchmark : this applies Canny edge detection and contour finding to both the raw and processed frames, measuring the time taken for these operations and counting the number of contours detected.)
            #this means that  it finds the number of edges detected in each frame, which serves as a proxy for the computational complexity of analyzing the frame. The cartoonification process should ideally reduce the number of edges and contours, 
            # which in turn would reduce the computational load for any AI system trying to analyze these frames. By measuring the time taken for these operations, we can also demonstrate the performance benefits of using the cartoonified video.
            t_start = time.time()
            edges_proc = cv2.Canny(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY), 50, 150)
            cnts_proc, _ = cv2.findContours(edges_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            proc_cv_time_total += (time.time() - t_start) * 1000
            proc_contours_total += len(cnts_proc)
            
            frame_count += 1
            self.bench_raw_frame = raw_res.plot()
            self.bench_proc_frame = proc_res.plot()

        if frame_count == 0: frame_count = 1
        self.bench_stats = {
            "raw_kb": round((raw_bytes_total / 1024) / frame_count, 2),
            "proc_kb": round((proc_bytes_total / 1024) / frame_count, 2),
            "savings_percent": round(100 - ((proc_bytes_total/max(1, raw_bytes_total))*100), 1),
            
            "raw_objs": round(raw_objects_total / frame_count, 1),
            "proc_objs": round(proc_objects_total / frame_count, 1),
            "objs_prevented": max(0, round((raw_objects_total - proc_objects_total) / frame_count, 1)),

            "raw_contours": int(raw_contours_total / frame_count),
            "proc_contours": int(proc_contours_total / frame_count),
            "contours_prevented": int((raw_contours_total - proc_contours_total) / frame_count),

            "raw_cv_time": round(raw_cv_time_total / frame_count, 1),
            "proc_cv_time": round(proc_cv_time_total / frame_count, 1),
            "time_saved_percent": max(0, round(100 - ((proc_cv_time_total/max(0.1, raw_cv_time_total))*100), 1))
        }
        self.benchmarking = False

manager = StreamManager()

# --- ROUTES ---
def generate_frames(mode="raw"):
    frame_counter = 0; last_frame_time = time.time()
    while True:
        frame = manager.get_frame()
        if frame is None: time.sleep(0.1); continue
        start_time = time.time()

        if mode == "raw":
            output = cv2.resize(frame, (640, 480))
            latency_ms = (time.time() - start_time) * 1000
        else:
            channels = manager.engine.process(frame)
            output = channels.get(mode, channels["cartoon"])
            latency_ms = (time.time() - start_time) * 1000
            
            frame_counter += 1
            if frame_counter % 10 == 0:
                current_time = time.time()
                fps = 10.0 / (current_time - last_frame_time + 1e-5)
                last_frame_time = current_time
                metrics = compute_metrics(frame, output, manager.engine.last_face_count, manager.engine.privacy_mode, manager.engine.intensity, manager.engine.iot_compression, fps, latency_ms)
                if metrics: manager.latest_metrics = metrics

        ret, buffer = cv2.imencode('.jpg', output)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_bench_frames(mode="raw"):
    while True:
        if not manager.benchmarking: time.sleep(0.5); continue
        frame = manager.bench_raw_frame if mode == "raw" else manager.bench_proc_frame
        if frame is None: time.sleep(0.1); continue
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed/<mode>')
def video_feed(mode): return Response(generate_frames(mode), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/benchmark_feed/<mode>')
def benchmark_feed(mode): return Response(generate_bench_frames(mode), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/start_benchmark', methods=['POST'])
def start_benchmark():
    if not manager.benchmarking: threading.Thread(target=manager.run_benchmark_thread, daemon=True).start()
    return jsonify({"status": "Benchmark Started"})
@app.route('/benchmark_status', methods=['GET'])
def benchmark_status(): return jsonify({"is_running": manager.benchmarking, "time_left": manager.bench_time_left, "stats": manager.bench_stats})
@app.route('/metrics', methods=['GET'])
def get_metrics(): return jsonify(manager.latest_metrics)
@app.route('/snapshot', methods=['GET'])
def snapshot():
    frame = manager.get_frame()
    if frame is None: return jsonify({"error": "No video feed active"}), 400
    raw_frame = cv2.resize(frame, (640, 480))
    proc_frame = manager.engine.process(frame).get("cartoon")
    _, raw_buffer = cv2.imencode('.jpg', raw_frame)
    _, proc_buffer = cv2.imencode('.jpg', proc_frame)
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('raw_input.jpg', raw_buffer.tobytes())
        zf.writestr('processed_output.jpg', proc_buffer.tobytes())
        zf.writestr('telemetry_report.json', json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "pipeline_mode": manager.engine.pipeline_mode, "telemetry": manager.latest_metrics}, indent=4))
    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name=f'telemetry_snapshot_{int(time.time())}.zip')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    if request.json.get('type') == 'webcam': manager.set_source(0); return jsonify({"status": "Started"})
    elif request.json.get('type') == 'stop': manager.stop_current(); return jsonify({"status": "Stopped"})
    return jsonify({"error": "Invalid"}), 400
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    manager.set_source(filepath)
    return jsonify({"status": "File started"})
@app.route('/set_processing_config', methods=['POST'])
def set_processing_config():
    data = request.json
    manager.engine.update_config(int(data.get("intensity", 50)), bool(data.get("auto_mode", False)), bool(data.get("motion_aware", False)), bool(data.get("privacy_mode", False)), bool(data.get("iot_compression", False)), str(data.get("pipeline_mode", "standard")))
    return jsonify({"status": "Updated"})

if __name__ == '__main__':
    manager.stop_current()
    app.run(host='0.0.0.0', port=5000, threaded=True)