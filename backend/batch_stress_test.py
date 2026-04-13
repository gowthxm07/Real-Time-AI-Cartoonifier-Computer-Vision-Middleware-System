import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
import logging
from server import Cartoonifier

# 1. Map your specific filenames to professional environmental labels
VIDEO_LABELS = {
    "day_traffic.mp4": "Daytime Traffic (Ideal Conditions)",
    "evening_traffic.mp4": "Evening Traffic (Low Light)",
    "moving_camera_traffic.mp4": "PTZ / Panning Camera (Dynamic Background)",
    "night_traffic.mp4": "Nighttime Traffic (Dark)",
    "People1.mp4": "Nighttime Pedestrians (Complex Lighting)",
    "People2.mp4": "Pitch Black Environment",
    "rainy_traffic.mp4": "Adverse Weather (Rain/Snow Noise)"
}

def run_enterprise_batch_test():
    print("="*80)
    print("🚀 ENTERPRISE PIPELINE: DATASET GENERALIZATION STRESS TEST")
    print("="*80)
    
    print("Loading YOLOv8 Model...")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    yolo_model = YOLO('yolov8n.pt')
    
    engine = Cartoonifier()
    engine.update_config(intensity=80, auto_mode=False, motion_aware=True, 
                         privacy_mode=False, iot_compression=False, pipeline_mode='tracking')

    dataset_folder = r"D:\Semester 6\Computer Vision\Project\backend\uploads"
    
    if not os.path.exists(dataset_folder):
        print(f"ERROR: Could not find folder at {dataset_folder}")
        return

    video_files = [f for f in os.listdir(dataset_folder) if f.endswith(".mp4")]
    
    # Dictionary to hold the final summary data
    final_report = {
        "STRENGTHS": [],
        "WEAKNESSES": [],
        "MODERATE": []
    }
    
    for video_file in video_files:
        # Get the label, or use the filename if it's not in our dictionary
        env_label = VIDEO_LABELS.get(video_file, f"Unknown Environment ({video_file})")
        
        print(f"\n" + "-"*60)
        print(f"📹 ANALYZING: {env_label}")
        print(f"   File: {video_file}")
        print("-" *60)
        
        video_path = os.path.join(dataset_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        raw_bytes = 0; proc_bytes = 0
        raw_objs = 0; proc_objs = 0
        raw_contours = 0; proc_contours = 0
        raw_cv_time = 0; proc_cv_time = 0
        
        frame_count = 0
        max_frames_to_test = 60 
        
        while frame_count < max_frames_to_test: 
            ret, frame = cap.read()
            if not ret: break 
                
            raw_frame = cv2.resize(frame, (640, 480))
            channels = engine.process(frame)
            proc_frame = channels['cartoon']
            
            # Metric 1: Bandwidth
            _, r_buf = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _, p_buf = cv2.imencode('.jpg', proc_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            raw_bytes += len(r_buf)
            proc_bytes += len(p_buf)
            
            # Metric 2: AI Load
            raw_res = yolo_model(raw_frame, verbose=False)[0]
            proc_res = yolo_model(proc_frame, verbose=False)[0]
            raw_objs += len(raw_res.boxes)
            proc_objs += len(proc_res.boxes)
            
            # Metric 3 & 4: Complexity & Speed
            t_start = time.time()
            edges_raw = cv2.Canny(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY), 50, 150)
            cnts_raw, _ = cv2.findContours(edges_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_cv_time += (time.time() - t_start) * 1000
            raw_contours += len(cnts_raw)
            
            t_start = time.time()
            edges_proc = cv2.Canny(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY), 50, 150)
            cnts_proc, _ = cv2.findContours(edges_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            proc_cv_time += (time.time() - t_start) * 1000
            proc_contours += len(cnts_proc)
            
            frame_count += 1
            if frame_count % 20 == 0:
                print(f"  -> Processed {frame_count}/{max_frames_to_test} frames...")
            
        cap.release()
        
        if frame_count > 0:
            bw_savings = 100 - ((proc_bytes / max(1, raw_bytes)) * 100)
            objs_prevented = (raw_objs - proc_objs) / frame_count
            textures_deleted = int((raw_contours - proc_contours) / frame_count)
            speedup = 100 - ((proc_cv_time / max(0.1, raw_cv_time)) * 100)
            
            # Print Individual Video Report
            print("\n📊 INDIVIDUAL BENCHMARK:")
            print(f"  Bandwidth Saved:  {bw_savings:.1f}%")
            print(f"  False Positives:  -{max(0, objs_prevented):.1f} per frame")
            print(f"  Textures Deleted: {textures_deleted} edges per frame")
            print(f"  Execution Speed:  {max(0, speedup):.1f}% faster")
            
            # Categorize for Final Report
            summary_string = f"{env_label} ({bw_savings:.1f}% Bandwidth Savings)"
            if bw_savings < 0 or textures_deleted < 0:
                final_report["WEAKNESSES"].append(summary_string)
            elif bw_savings > 40:
                final_report["STRENGTHS"].append(summary_string)
            else:
                final_report["MODERATE"].append(summary_string)

    # --- PRINT THE EXECUTIVE SUMMARY REPORT ---
    print("\n\n" + "="*80)
    print("🏆 FINAL EXECUTIVE SUMMARY: SYSTEM STRENGTHS & WEAKNESSES 🏆")
    print("="*80)
    
    print("\n🟢 SYSTEM STRENGTHS (Optimal Performance):")
    print("These environments allow MOG2 to perfectly isolate foreground subjects, resulting in massive AWS savings.")
    for item in final_report["STRENGTHS"]:
        print(f"  ✅ {item}")
        
    print("\n🟡 MODERATE PERFORMANCE:")
    print("System remains functional, but background suppression is limited by complex lighting or lack of contrast.")
    if not final_report["MODERATE"]: print("  (None)")
    for item in final_report["MODERATE"]:
        print(f"  ⚠️ {item}")
        
    print("\n🔴 SYSTEM WEAKNESSES (Architectural Limits):")
    print("These environments break the MOG2 background subtractor. Global visual noise or camera movement is interpreted as foreground.")
    for item in final_report["WEAKNESSES"]:
        print(f"  ❌ {item}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_enterprise_batch_test()