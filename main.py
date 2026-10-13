import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
import threading 

stop = threading.Event()

def record_audio(label):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate = 16000, input = True, frames_per_buffer=1024)

    frames = []
    stop.clear()
    print(f"\n{label} - press enter to stop")
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    while not stop.is_set():
        frames.append(stream.read(1024, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.frombuffer(b"".join(frames), dtype = np.int16)
def analyse(samples):
    duration = len(samples) / 16000
    volume = np.mean(np.abs(samples)) if len(samples) else 0
    try:
        audio = sr.AudioData(samples.astype(np.int16).tobytes(),16000, 2)
        text = sr.Recognizer().recognize_google(audio)
    except Exception:
        text = "could not transcribe"
    print(f"duration: {duration:.2f} seconds")
    print(f"Average Volume: {volume:.0f}")
    print(f"Speech: {text}")
    return duration, volume
audio1 = record_audio("Recording 1: Speak normally")
stats1 = analyse(audio1)
input("\nPress enter for the second recording")
audio2 = record_audio("recording 2: speak lowder or faster")
stats2 = analyse(audio2)
print("longer recording:", 1 if stats1[0] > stats2[0] else 2)
print("louder recording:", 1 if stats1[1] > stats2[1] else 2)

plt.figure(figsize=(10, 5))
plt.subplot(2,1,1)
plt.plot(np.arange(len(audio1)) / 16000, audio1)
plt.title("Recording 1")

plt.subplot(2,1,2)
plt.plot(np.arange(len(audio2)) / 16000, audio2)
plt.title("Recording 2")
plt.xlabel("time (seconds)")

plt.tight_layout()
plt.show()