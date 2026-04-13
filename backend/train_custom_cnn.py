import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from server import Cartoonifier

def build_minimal_cnn():
    """Builds a small CNN with Dropout to prevent Overfitting."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5), # <--- NEW: Forces the AI to generalize
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_training_experiment(video_path=r'D:\Semester 6\Computer Vision\Project\backend\traffic.mp4', max_frames=600):
    print("="*50)
    print("🚀 EXTRACTING & AUTO-LABELING DATA...")
    print("="*50)
    
    engine = Cartoonifier()
    engine.update_config(intensity=80, auto_mode=False, motion_aware=True, 
                         privacy_mode=False, iot_compression=False, pipeline_mode='tracking')

    cap = cv2.VideoCapture(video_path)
    
    raw_dataset = []
    proc_dataset = []
    motion_counts = []
    
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
            
        channels = engine.process(frame)
        proc_frame = channels['cartoon']
        
        mask = channels['mask']
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        motion_pixels = cv2.countNonZero(gray_mask)
        
        raw_small = cv2.resize(frame, (64, 64)) / 255.0 
        proc_small = cv2.resize(proc_frame, (64, 64)) / 255.0
        
        raw_dataset.append(raw_small)
        proc_dataset.append(proc_small)
        motion_counts.append(motion_pixels)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{max_frames} frames...")

    cap.release()
    
    median_motion = np.median(motion_counts)
    labels = [1 if m > median_motion else 0 for m in motion_counts]
    
    X_raw = np.array(raw_dataset)
    X_proc = np.array(proc_dataset)
    Y = np.array(labels)
    
    # --- NEW: RANDOM SHUFFLE TO PREVENT SEQUENCE BIAS ---
    print("\n🔀 Shuffling dataset to prevent chronological overfitting...")
    indices = np.arange(X_raw.shape[0])
    np.random.shuffle(indices)
    X_raw = X_raw[indices]
    X_proc = X_proc[indices]
    Y = Y[indices]
    
    print(f"Dataset built! Total frames: {len(Y)} | Heavy Traffic (1): {sum(Y)} | Light Traffic (0): {len(Y)-sum(Y)}")

    # --- TRAIN MODEL A (RAW) ---
    print("\n" + "="*50)
    print("🧠 TRAINING MODEL A: RAW VIDEO CLASSIFIER")
    print("="*50)
    model_raw = build_minimal_cnn()
    start_time = time.time()
    history_raw = model_raw.fit(X_raw, Y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    raw_train_time = time.time() - start_time

    # --- TRAIN MODEL B (PROCESSED) ---
    print("\n" + "="*50)
    print("🧠 TRAINING MODEL B: PROCESSED VIDEO CLASSIFIER")
    print("="*50)
    model_proc = build_minimal_cnn()
    start_time = time.time()
    history_proc = model_proc.fit(X_proc, Y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    proc_train_time = time.time() - start_time

    # --- SHOW BENCHMARK RESULTS ---
    print("\n" + "="*50)
    print("🏆 CUSTOM CNN TRAINING BENCHMARK REPORT 🏆")
    print("="*50)
    
    raw_final_acc = history_raw.history['val_accuracy'][-1] * 100
    proc_final_acc = history_proc.history['val_accuracy'][-1] * 100
    
    print(f"Model A (Raw Video) Final Accuracy:       {raw_final_acc:.2f}% (Took {raw_train_time:.1f} sec)")
    print(f"Model B (Processed Video) Final Accuracy: {proc_final_acc:.2f}% (Took {proc_train_time:.1f} sec)")
    
    # Plotting the Graph!
    plt.figure(figsize=(10, 5))
    plt.plot(history_raw.history['val_accuracy'], label='Raw Video (Validation Acc)', color='red', linestyle='dashed', linewidth=2)
    plt.plot(history_proc.history['val_accuracy'], label='Processed Video (Validation Acc)', color='green', linewidth=3)
    plt.title('Custom CNN Training: Heavy vs Light Traffic Classification')
    plt.xlabel('Epochs (Training Cycles)')
    plt.ylabel('AI Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cnn_benchmark_graph.png')
    print("\n✅ A comparison graph has been saved as 'cnn_benchmark_graph.png' in your folder!")
    plt.show()

if __name__ == "__main__":
    run_training_experiment()