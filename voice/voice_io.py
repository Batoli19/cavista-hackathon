import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
import threading
import time
import whisper
import tkinter as tk
from tkinter import ttk
import pandas as pd

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
CHANNELS = 1
FILENAME = "consultation.wav"

recording = False
audio_frames = []
start_time = None

# Load Whisper multilingual model
model = whisper.load_model("base")

# ==============================
# LOAD THOUSANDS OF DISEASES
# ==============================
# CSV format example: disease_name,keywords
# keywords = comma-separated terms including English and Setswana
# e.g., "Migraine,headache,tlhogo"
diseases_df = pd.read_csv("icd10_diseases.csv")
diseases_df['keywords'] = diseases_df['keywords'].apply(lambda x: [k.strip().lower() for k in x.split(',')])

# ==============================
# ADVANCED DIAGNOSIS FUNCTION
# ==============================
def diagnose_advanced(text):
    """
    Doctor-level diagnosis engine.
    Matches transcript keywords to thousands of diseases dynamically.
    """
    text = text.lower()
    matched_diseases = []

    for _, row in diseases_df.iterrows():
        for kw in row['keywords']:
            if kw in text:
                matched_diseases.append(row['disease_name'])
                break  # Avoid duplicate matches per disease

    if not matched_diseases:
        return "No clear diagnosis detected. Recommend further medical evaluation."

    # Prioritize severe conditions (heart attack, stroke, cancer, etc.)
    severity_order = [
        "Heart attack", "Stroke", "Cancer", "Brain tumor", "Pulmonary embolism",
        "Sepsis", "Kidney failure", "Diabetes"
    ]

    sorted_diseases = sorted(
        matched_diseases,
        key=lambda x: severity_order.index(x) if x in severity_order else len(severity_order) + 1
    )

    return ", ".join(sorted_diseases)

# ==============================
# RECORDING ENGINE
# ==============================
def record_audio():
    global recording, audio_frames

    def callback(indata, frames, time_info, status):
        if recording:
            audio_frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        callback=callback):
        while recording:
            sd.sleep(100)

def start_recording():
    global recording, audio_frames, start_time
    recording = True
    audio_frames = []
    start_time = time.time()
    indicator_label.config(text="🟢 Listening", foreground="green")
    threading.Thread(target=record_audio, daemon=True).start()
    update_timer()

def stop_recording():
    global recording
    recording = False
    indicator_label.config(text="🔴 Stopped", foreground="red")
    save_audio()

# ==============================
# SAVE + PROCESS AUDIO
# ==============================
def save_audio():
    global audio_frames

    if not audio_frames:
        return

    audio = np.concatenate(audio_frames, axis=0)

    # Noise reduction
    reduced = nr.reduce_noise(
        y=audio.flatten(),
        sr=SAMPLE_RATE
    )

    wav.write(FILENAME, SAMPLE_RATE, reduced)
    process_audio(FILENAME)

# ==============================
# TRANSCRIPTION + DIAGNOSIS
# ==============================
def process_audio(file_path):
    print("Transcribing...")
    result = model.transcribe(file_path)  # Auto-detect language
    transcript = result["text"]

    transcript_box.delete("1.0", tk.END)
    transcript_box.insert(tk.END, transcript)

    diagnosis = diagnose_advanced(transcript)
    diagnosis_box.delete("1.0", tk.END)
    diagnosis_box.insert(tk.END, diagnosis)

# ==============================
# TIMER
# ==============================
def update_timer():
    if recording:
        elapsed = int(time.time() - start_time)
        mins = elapsed // 60
        secs = elapsed % 60
        timer_label.config(text=f"{mins:02d}:{secs:02d}")
        root.after(1000, update_timer)

# ==============================
# UI
# ==============================
root = tk.Tk()
root.title("Professional Clinical Voice Consultation System")
root.geometry("700x550")

title = ttk.Label(root, text="Professional Consultation Voice System", font=("Arial", 16))
title.pack(pady=10)

indicator_label = ttk.Label(root, text="🔴 Idle", font=("Arial", 12))
indicator_label.pack()

timer_label = ttk.Label(root, text="00:00", font=("Arial", 18))
timer_label.pack(pady=5)

start_btn = ttk.Button(root, text="Start Consultation", command=start_recording)
start_btn.pack(pady=5)

stop_btn = ttk.Button(root, text="Stop Consultation", command=stop_recording)
stop_btn.pack(pady=5)

ttk.Label(root, text="Transcript").pack()
transcript_box = tk.Text(root, height=8)
transcript_box.pack()

ttk.Label(root, text="AI Preliminary Diagnosis").pack()
diagnosis_box = tk.Text(root, height=6)
diagnosis_box.pack()

root.mainloop()
