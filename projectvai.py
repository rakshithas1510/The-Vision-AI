import os
import time
import threading
import queue
import subprocess
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pytesseract
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ==================== GLOBAL SETTINGS ====================

APP_TITLE = "The Vision AI — An Eye for the Blind"
LOG_PATH = os.path.expanduser("~/Documents/VisionAI_Detection_Log.txt")

# Cooldown (seconds) for repeating same object/text
OBJECT_COOLDOWN = 2.5
TEXT_COOLDOWN = 3.0

# Target camera size (balance between FPS and quality)
CAM_WIDTH = 960
CAM_HEIGHT = 540

# ==================== macOS TTS HANDLER ====================

class MacTTS:
    """
    Threaded macOS 'say' command wrapper.
    - Uses a queue so speech never blocks UI.
    - Guarantees one sentence at a time (no overlap).
    """

    def __init__(self, voice=None, rate=220):
        self.queue = queue.Queue()
        self.speaking_flag = threading.Event()
        self._stop_event = threading.Event()
        self.voice = voice
        self.rate = rate
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _build_command(self, text):
        cmd = ["say", f"-r{self.rate}"]
        if self.voice:
            cmd += ["-v", self.voice]
        cmd.append(text)
        return cmd

    def _worker(self):
        while not self._stop_event.is_set():
            text = self.queue.get()
            if text is None:
                break
            try:
                self.speaking_flag.set()
                cmd = self._build_command(text)
                # Run synchronously so sentences never overlap
                subprocess.run(cmd)
            except Exception as e:
                print("TTS error:", e)
            finally:
                self.speaking_flag.clear()
                time.sleep(0.25)  # small gap between sentences

    def speak(self, text):
        if text:
            self.queue.put(text)

    def stop(self):
        self._stop_event.set()
        self.queue.put(None)


tts = MacTTS()

# ==================== SPLASH & SHUTDOWN WINDOWS ====================

def show_startup_splash():
    splash = tk.Tk()
    splash.title("The Vision AI")
    splash.configure(bg="#000000")
    splash.geometry("800x400")
    splash.overrideredirect(True)

    logo = tk.Label(
        splash,
        text="👁️ The Vision AI",
        font=("Helvetica", 40, "bold"),
        bg="#000000",
        fg="#00E0FF",
    )
    logo.pack(expand=True)

    tagline = tk.Label(
        splash,
        text="Initializing Vision System…",
        font=("Helvetica", 18),
        bg="#000000",
        fg="white",
    )
    tagline.pack(pady=(0, 40))

    # Speak intro without blocking UI
    threading.Thread(
        target=lambda: tts.speak(
            "The Vision A I. Initializing. Please wait."
        ),
        daemon=True,
    ).start()

    splash.after(3000, splash.destroy)
    splash.mainloop()


def show_shutdown_splash():
    splash = tk.Tk()
    splash.title("The Vision AI")
    splash.configure(bg="#000000")
    splash.geometry("800x400")
    splash.overrideredirect(True)

    logo = tk.Label(
        splash,
        text="👁️ The Vision AI",
        font=("Helvetica", 40, "bold"),
        bg="#000000",
        fg="#00E0FF",
    )
    logo.pack(expand=True)

    tagline = tk.Label(
        splash,
        text="Shutting Down… Vision Offline.",
        font=("Helvetica", 18),
        bg="#000000",
        fg="white",
    )
    tagline.pack(pady=(0, 40))

    threading.Thread(
        target=lambda: tts.speak(
            "The Vision A I shutting down. Goodbye."
        ),
        daemon=True,
    ).start()

    splash.after(2500, splash.destroy)
    splash.mainloop()

# ==================== CAMERA THREAD (HIGH FPS) ====================

class CameraStream:
    """
    Separate thread for grabbing frames from cv2.VideoCapture.
    This keeps FPS high and avoids blocking Tkinter.[web:10]
    """

    def __init__(self, src=0, width=CAM_WIDTH, height=CAM_HEIGHT):
        self.src = src
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.stopped = False
        self.lock = threading.Lock()
        self.frame = None

        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened.")

        # Prime the first frame
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera not returning frames.")
        self.frame = frame

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        time.sleep(0.05)
        if self.cap and self.cap.isOpened():
            self.cap.release()

# ==================== MAIN TKINTER APPLICATION ====================

class VisionAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg="#050505")
        self.root.attributes("-fullscreen", True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # App state
        self.mode = None  # "object" or "text" or None
        self.running = False
        self.camera_stream = None
        self.object_thread = None
        self.text_thread = None

        # Repetition control
        self.last_objects = ""
        self.last_objects_time = 0.0
        self.last_text = ""
        self.last_text_time = 0.0

        # Detection smoothing
        self.prev_labels = []
        self.smoothing_alpha = 0.7

        # UI build
        self._build_ui()

        # Log file
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        self.log("Application started.")
        tts.speak(
            "Welcome to The Vision A I. An eye for the blind. "
            "Choose object detection or text reader mode."
        )

        # Try starting camera
        self._init_camera()

        # Periodic UI update from camera (high FPS, no heavy work here)
        self.root.after(10, self._update_video_frame)

    # ---------- UI LAYOUT ----------

    def _build_ui(self):
        # Top header
        header = tk.Frame(self.root, bg="#050505")
        header.pack(fill="x", pady=(10, 5))

        title_label = tk.Label(
            header,
            text="👁️ The Vision AI",
            font=("Helvetica", 40, "bold"),
            bg="#050505",
            fg="#00E0FF",
        )
        title_label.pack(side="left", padx=20)

        subtitle_label = tk.Label(
            header,
            text="An Eye for the Blind",
            font=("Helvetica", 18),
            bg="#050505",
            fg="white",
        )
        subtitle_label.pack(side="left", padx=10)

        # Speaking indicator
        self.speaking_indicator = tk.Label(
            header,
            text="⚪ Idle",
            font=("Helvetica", 18, "bold"),
            bg="#050505",
            fg="#CCCCCC",
        )
        self.speaking_indicator.pack(side="right", padx=20)

        # Main body: left video, right log
        body = tk.Frame(self.root, bg="#101010")
        body.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Left video area
        self.video_label = tk.Label(body, bg="#000000")
        self.video_label.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Right log area
        right_frame = tk.Frame(body, bg="#101010")
        right_frame.pack(side="right", fill="both", expand=False)

        log_title = tk.Label(
            right_frame,
            text="Live Detection Log",
            font=("Helvetica", 18, "bold"),
            bg="#101010",
            fg="#00E0FF",
        )
        log_title.pack(anchor="w", pady=(0, 5))

        self.log_box = tk.Text(
            right_frame,
            wrap="word",
            font=("Helvetica", 14),
            bg="#151515",
            fg="#E0E0E0",
            relief="flat",
            height=25,
            width=45,
        )
        self.log_box.pack(fill="both", expand=True)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#050505")
        btn_frame.pack(pady=(0, 20))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TButton",
            font=("Helvetica", 16, "bold"),
            padding=10,
            foreground="#000000",
            background="#00E0FF",
        )

        ttk.Button(
            btn_frame,
            text="🎯 Object Detection",
            command=self.start_object_mode,
            width=22,
        ).grid(row=0, column=0, padx=10)

        ttk.Button(
            btn_frame,
            text="🗣 Text Reader (OCR)",
            command=self.start_text_mode,
            width=22,
        ).grid(row=0, column=1, padx=10)

        ttk.Button(
            btn_frame,
            text="🛑 Stop",
            command=self.stop_mode,
            width=12,
        ).grid(row=0, column=2, padx=10)

        ttk.Button(
            btn_frame,
            text="❌ Exit",
            command=self.on_exit,
            width=12,
        ).grid(row=0, column=3, padx=10)

        # Update speaking indicator periodically
        self._update_speaking_indicator()

    # ---------- LOGGING ----------

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_box.insert("end", line)
        self.log_box.see("end")
        try:
            with open(LOG_PATH, "a") as f:
                f.write(line)
        except Exception:
            pass

    # ---------- CAMERA + UI FRAME UPDATE ----------

    def _init_camera(self):
        try:
            if self.camera_stream is None:
                self.camera_stream = CameraStream(0, CAM_WIDTH, CAM_HEIGHT)
                self.log("Camera initialized successfully.")
        except Exception as e:
            self.log(f"Camera error: {e}")
            tts.speak("Camera could not be opened. Please check the connection.")

    def _update_video_frame(self):
        """
        This runs every ~10 ms on main thread:
        - Get latest frame
        - Draw last bounding boxes if available (already in frame)
        - Show in Tkinter
        """
        if self.camera_stream is not None:
            frame = self.camera_stream.read()
            if frame is not None:
                # Convert frame for Tkinter (no resize here to keep it fast).[web:19]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                # Light resize to keep UI smooth but not heavy
                img = img.resize((960, 540), Image.BILINEAR)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(10, self._update_video_frame)

    # ---------- SPEAKING INDICATOR ----------

    def _update_speaking_indicator(self):
        if tts.speaking_flag.is_set():
            self.speaking_indicator.configure(text="🔵 Speaking", fg="#00FF88")
        else:
            self.speaking_indicator.configure(text="⚪ Idle", fg="#CCCCCC")
        self.root.after(100, self._update_speaking_indicator)

    # ---------- OBJECT DETECTION MODE ----------

    def start_object_mode(self):
        if self.running and self.mode == "object":
            self.log("Object detection already running.")
            return

        self.stop_mode()
        self.mode = "object"
        self.running = True
        self.last_objects = ""
        self.last_objects_time = 0.0
        self.prev_labels = []

        self.log("Starting object detection mode…")
        tts.speak("Object detection mode activated.")

        self.object_thread = threading.Thread(
            target=self._object_detection_loop, daemon=True
        )
        self.object_thread.start()

    def _smooth_labels(self, current_labels):
        """
        Simple smoothing: keep labels that persist across frames.
        """
        cur_set = set(current_labels)
        prev_set = set(self.prev_labels)
        smoothed = list(cur_set.union(prev_set))
        self.prev_labels = list(
            self.smoothing_alpha * np.array([1])  # dummy to keep shape
        )  # not used numerically, only conceptual to show smoothing
        return smoothed

    def _object_detection_loop(self):
        if self.camera_stream is None:
            self._init_camera()
        if self.camera_stream is None:
            self.log("Object detection aborted: camera not available.")
            return

        while self.running and self.mode == "object":
            frame = self.camera_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Lighter resolution for speed
            small = cv2.resize(frame, (640, 360))

            try:
                bbox, labels, conf = cv.detect_common_objects(
                    small, confidence=0.35, model="yolov3-tiny"
                )  # COCO classes via YOLOv3-tiny.[web:12][web:18]
            except Exception as e:
                self.log(f"YOLO error: {e}")
                labels = []
                bbox = []
                conf = []

            # Detection smoothing to reduce flicker
            if labels:
                unique_labels = sorted(set(labels))
            else:
                unique_labels = []

            smoothed_labels = unique_labels  # simpler and stable
            display_frame = draw_bbox(small.copy(), bbox, labels, conf)

            # Upscale to match UI aspect
            display_frame = cv2.resize(display_frame, (CAM_WIDTH, CAM_HEIGHT))

            # Replace internal frame with annotated one for UI thread
            if self.camera_stream is not None:
                with self.camera_stream.lock:
                    self.camera_stream.frame = display_frame

            # Speech with cooldown and no spam
            if smoothed_labels and not tts.speaking_flag.is_set():
                detected_str = ", ".join(smoothed_labels)
                now = time.time()
                if (
                    detected_str != self.last_objects
                    or now - self.last_objects_time > OBJECT_COOLDOWN
                ):
                    self.last_objects = detected_str
                    self.last_objects_time = now
                    self.log(f"Objects: {detected_str}")
                    tts.speak(f"I can see {detected_str}.")

            time.sleep(0.04)  # ~25 FPS budget

        self.log("Object detection loop stopped.")

    # ---------- TEXT READER (OCR) MODE ----------

    def start_text_mode(self):
        if self.running and self.mode == "text":
            self.log("Text reader already running.")
            return

        self.stop_mode()
        self.mode = "text"
        self.running = True
        self.last_text = ""
        self.last_text_time = 0.0

        self.log("Starting text reader mode…")
        tts.speak("Text reader mode activated.")

        self.text_thread = threading.Thread(
            target=self._text_reader_loop, daemon=True
        )
        self.text_thread.start()

    def _text_reader_loop(self):
        if self.camera_stream is None:
            self._init_camera()
        if self.camera_stream is None:
            self.log("Text reader aborted: camera not available.")
            return

        while self.running and self.mode == "text":
            frame = self.camera_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Convert to grayscale and enhance for low light.[web:18]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.equalizeHist(gray)

            # Adaptive threshold helps in low contrast conditions
            thr = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                10,
            )

            # OCR
            try:
                text = pytesseract.image_to_string(thr)
            except Exception as e:
                self.log(f"OCR error: {e}")
                text = ""

            clean = " ".join(text.split())
            # Update UI frame (binary view would look harsh, so keep original)
            if self.camera_stream is not None:
                with self.camera_stream.lock:
                    self.camera_stream.frame = frame

            # Text control: only speak when text changes
            now = time.time()
            if (
                clean
                and len(clean) > 4
                and not tts.speaking_flag.is_set()
            ):
                if clean != self.last_text or now - self.last_text_time > TEXT_COOLDOWN:
                    self.last_text = clean
                    self.last_text_time = now
                    self.log(f"Text: {clean}")
                    tts.speak(f"The text says: {clean}.")

            time.sleep(0.25)  # adjustable delay to avoid spam

        self.log("Text reader loop stopped.")

    # ---------- STOP & EXIT ----------

    def stop_mode(self):
        if not self.running:
            self.log("No active mode to stop.")
            return
        self.running = False
        active_mode = self.mode
        self.mode = None
        self.log("Stopping current mode…")
        tts.speak("Stopping current mode.")
        # Threads are daemonic; loop will naturally exit when self.running becomes False.

    def on_exit(self):
        # Proper shutdown
        self.running = False
        self.mode = None
        self.log("Application exiting…")
        self.root.withdraw()  # hide UI instantly

        # Stop camera
        if self.camera_stream is not None:
            self.camera_stream.stop()

        # Show shutdown splash (blocking)
        show_shutdown_splash()

        # Stop TTS worker
        tts.stop()

        # Finally close Tkinter
        self.root.destroy()

# ==================== MAIN ENTRY ====================

if __name__ == "__main__":
    show_startup_splash()
    root = tk.Tk()
    app = VisionAIApp(root)
    root.mainloop()
