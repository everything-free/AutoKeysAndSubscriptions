import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
import soundcard as sc
import json
import time
from vosk import Model, KaldiRecognizer

MODEL_PATH = "model"
KEYWORD = "ключ"
VIDEO_FILE = "video.mov"
SAMPLE_RATE = 16000
LEV_THRESHOLD = 3
COOLDOWN = 15.0

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            add = prev[j] + 1
            delete = cur[j - 1] + 1
            change = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(add, delete, change)
        prev = cur
    return prev[lb]

def quick_match(text: str, keyword: str, max_dist: int) -> bool:
    text = text.lower()
    keyword = keyword.lower()
    if keyword.startswith(text) and len(text) >= 2:
        return True
    if text.startswith(keyword[:2]):
        return True
    if levenshtein(text, keyword) <= max_dist:
        return True
    return False

class VideoWindow(QMainWindow):
    def __init__(self, video_path: Path):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.label)

        self.frames = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        print("Предобработка видео... Это может занять время.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            lower_green = np.array([60, 60, 60])
            upper_green = np.array([80, 255, 255])
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            alpha = cv2.bitwise_not(mask)
            frame_rgba[:, :, 3] = alpha


            h, w, ch = frame_rgba.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
            self.frames.append(qimg)

        cap.release()
        if not self.frames:
            raise RuntimeError("Видео пустое")

        self.total_frames = len(self.frames)
        print(f"Загружено {self.total_frames} кадров.")

        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.player.setSource(QUrl.fromLocalFile(str(video_path)))
        self.player.mediaStatusChanged.connect(self.handle_media_status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.window_w = QApplication.primaryScreen().size().width()
        self.window_h = QApplication.primaryScreen().size().height()

    def start_playback(self):
        self.showFullScreen()
        self.player.setPosition(0)
        self.player.play()
        self.timer.start(10)

    def update_frame(self):
        if self.player.duration() == 0:
            return
        pos_ms = self.player.position()
        frame_time = pos_ms / 1000.0
        frame_num = int(frame_time * self.fps) % self.total_frames
        if hasattr(self, 'last_frame') and frame_num == self.last_frame:
            return
        self.last_frame = frame_num
        qimg = self.frames[frame_num]
        self.label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.window_w, self.window_h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))

    def handle_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.stop()
            self.timer.stop()
            self.hide()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, ev):
        self.player.stop()
        self.timer.stop()
        ev.accept()

class ListenerThread(QThread):
    trigger = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model = Model(MODEL_PATH)
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)

        default_speaker = sc.default_speaker()
        self.mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)

        print("Слушаю системный звук...")

    def run(self):
        last_trigger = 0.0
        with self.mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
            while True:
                data = recorder.record(numframes=2000)
                data16 = (data * 32767).astype(np.int16).tobytes()

                triggered = False

                pres = self.rec.PartialResult()
                j = json.loads(pres)
                part = j.get("partial", "")
                if part and quick_match(part, KEYWORD, LEV_THRESHOLD):
                    triggered = True

                if self.rec.AcceptWaveform(data16):
                    res = self.rec.Result()
                    j = json.loads(res)
                    text = j.get("text", "")
                    if text and quick_match(text, KEYWORD, LEV_THRESHOLD):
                        triggered = True

                if triggered:
                    now = time.time()
                    if now - last_trigger >= COOLDOWN:
                        print(f"Найдено '{KEYWORD}' (или его часть). Включаю видео!")
                        self.trigger.emit()
                        last_trigger = now

def main():
    app = QApplication(sys.argv)
    base_dir = Path(__file__).parent
    video_path = base_dir / VIDEO_FILE
    if not video_path.exists():
        print(f"Файл {VIDEO_FILE} не найден!")
        sys.exit(1)

    video_win = VideoWindow(video_path)

    listener = ListenerThread()
    listener.trigger.connect(video_win.start_playback)
    listener.start()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
