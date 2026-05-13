"""
Jarvis - 개인 AI 비서 (웨이크워드 + 대화 + 제스처 + 모션)

─── 음성 ────────────────────────────────────────────────
  "자비스" 또는 "jarvis" → 대화 모드 (45초 유지)
  대화 중 → Claude Sonnet과 다중 턴 대화
  "잘 있어" / "종료" → 슬립 모드
  Siri: "Hey Siri, 자비스 시작해" → start_jarvis.sh 실행

─── 제스처 ──────────────────────────────────────────────
  ✋ 펼친 손  (0.8s) → 창 최대화
  ✊ 주먹     (0.8s) → 창 최소화
  ✌️ V사인   (0.8s) → 창 복원
  ☝️ 검지만   → 창 실시간 이동

h: 도움말  |  q: 종료
"""

import cv2
import math
import re
import time
import os
import base64
import queue
import threading
import subprocess
import numpy as np
import speech_recognition as sr
import sounddevice as sd
import whisper as _whisper
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

import os as _os
_os.environ.setdefault("GLOG_minloglevel", "3")

# ── 모션 감지 ──────────────────────────────────────────
MOTION_MIN_AREA       = 800
MOTION_MAX_RATIO      = 0.35
MOTION_ASPECT_MAX     = 7.0
MOTION_SOLIDITY_MIN   = 0.35
MOTION_DIFF_PIXELS    = 500
MOTION_PERSIST_FRAMES = 5
MOTION_CONFIRM_RATIO  = 0.6
MOTION_COOLDOWN       = 10
# ── 제스처 / 창 제어 ────────────────────────────────────
GESTURE_DWELL    = 0.8
GESTURE_COOLDOWN = 2.0
MOVE_INTERVAL    = 0.06
MOVE_SMOOTH      = 0.22
CAM_MARGIN       = 0.12
# ── 대화 / 웨이크워드 ──────────────────────────────────
WAKE_WORDS    = ["자비스", "jarvis"]
SLEEP_WORDS   = ["잘 있어", "자비스 꺼", "대화 종료", "슬립 모드", "종료할게"]
AWAKE_TIMEOUT = 45       # 대화 모드 유지 시간 (초)
MAX_HISTORY   = 10       # 최대 대화 턴 수 (왕복)
CONV_DISPLAY  = 3        # 화면에 표시할 최근 대화 수
SYSTEM_PROMPT = (
    "당신은 자비스(JARVIS)입니다. 사용자의 스마트하고 유능한 한국어 개인 AI 비서입니다. "
    "항상 한국어로 대화하며, 2~3문장으로 간결하게 답변합니다. "
    "카메라 영상을 통해 주변 상황을 파악할 수 있고, 이전 대화 맥락을 기억합니다. "
    "친근하고 지적인 어조로 사용자를 돕고, 필요 시 유머도 섞어주세요."
)
# ── 음성 ───────────────────────────────────────────────
SAMPLE_RATE       = 16000
CHUNK_SIZE        = 1024
SILENCE_THRESHOLD = 600
SILENCE_DURATION  = 1.2
MIN_SPEECH_CHUNKS = 8
# ──────────────────────────────────────────────────────

conv_history: list = []   # Claude 다중 턴 대화 히스토리

# Whisper 모델 로드 (base: 속도/정확도 균형, small: 한국어 정확도 향상)
print("[Jarvis] Whisper 모델 로딩 중...")
_whisper_model = _whisper.load_model("small")
print("[Jarvis] Whisper 준비 완료")

state = {
    "lock":        threading.Lock(),
    "frame":       None,
    "analysis":    "",
    "voice_text":  "",
    "listening":   False,
    "speaking":    False,
    "motion":      False,
    "new_command": False,
    "awake":       False,
    "awake_until": 0.0,
    "conv_log":    [],    # [(role, text), ...] 화면 표시용
}

HELP_TEXT = [
    "[ 음성 명령 ]",
    "  '자비스' → 대화 모드 진입",
    "  '잘 있어' → 슬립 모드",
    "",
    "[ 제스처 ]",
    "  ✋ 펼친 손  → 창 최대화  (0.8s)",
    "  ✊ 주먹     → 창 최소화  (0.8s)",
    "  ✌️ V사인   → 창 복원    (0.8s)",
    "  ☝️ 검지     → 창 이동",
    "",
    "[ 키보드 ]",
    "  h  도움말  |  q  종료",
]


# ── 유틸 ────────────────────────────────────────────────

def get_screen_size():
    try:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "Finder" to get bounds of window of desktop'],
            capture_output=True, text=True, timeout=2,
        )
        p = r.stdout.strip().split(", ")
        return int(p[2]), int(p[3])
    except Exception:
        return 1440, 900


# ── 정밀 모션 감지 ──────────────────────────────────────

class MotionDetector:
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=True
        )
        self.k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.prev_gray  = None
        self.history    = []
        self.confidence = 0.0

    def update(self, frame):
        h, w       = frame.shape[:2]
        frame_area = h * w
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised   = cv2.bilateralFilter(gray, 9, 75, 75)

        lr = 0.002 if self.confidence > 0.5 else 0.006
        fg = self.bg_sub.apply(denoised, learningRate=lr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.k_open,  iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close, iterations=2)

        has_prev   = self.prev_gray is not None
        diff_valid = False
        if has_prev:
            diff        = cv2.absdiff(self.prev_gray, gray)
            _, diff_bin = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            diff_valid  = cv2.countNonZero(diff_bin) > MOTION_DIFF_PIXELS
        self.prev_gray = gray

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MOTION_MIN_AREA or area > frame_area * MOTION_MAX_RATIO:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if max(bw, bh) / max(min(bw, bh), 1) > MOTION_ASPECT_MAX:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area > 0 and (area / hull_area) < MOTION_SOLIDITY_MIN:
                continue
            valid.append(cnt)

        detected = bool(valid) and (not has_prev or diff_valid)
        self.history.append(detected)
        if len(self.history) > MOTION_PERSIST_FRAMES:
            self.history.pop(0)
        self.confidence = sum(self.history) / max(len(self.history), 1)
        return valid, self.confidence >= MOTION_CONFIRM_RATIO, self.confidence


# ── 손 제스처 감지 (MediaPipe Tasks API 0.10+) ──────────

class HandGestureDetector:
    _CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    PATTERNS = {
        "OPEN_PALM": (True,  True,  True,  True,  True),
        "FIST":      (False, False, False, False, False),
        "POINT":     (False, True,  False, False, False),
        "PEACE":     (False, True,  True,  False, False),
        "THUMBS_UP": (True,  False, False, False, False),
    }
    GESTURE_KO = {
        "OPEN_PALM": "펼친 손", "FIST": "주먹",
        "POINT": "검지 이동",   "PEACE": "V사인",
        "THUMBS_UP": "엄지척",  "UNKNOWN": "?",
    }

    def __init__(self, model_path: str):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.72,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.time()

    def _finger_states(self, lm, handedness):
        is_right = (handedness == "Right")
        thumb_up = (lm[4].x < lm[3].x) if is_right else (lm[4].x > lm[3].x)
        fingers  = [thumb_up]
        for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
            fingers.append(lm[tip].y < lm[pip].y)
        return tuple(fingers)

    def _draw_skeleton(self, frame, lm):
        h, w = frame.shape[:2]
        pts  = [(int(l.x * w), int(l.y * h)) for l in lm]
        for a, b in self._CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 210, 0), 2)
        for i, (px, py) in enumerate(pts):
            r = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, (px, py), r, (0, 255, 180), -1)

    def detect(self, frame):
        h, w  = frame.shape[:2]
        ts_ms = int((time.time() - self._t0) * 1000)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res   = self.landmarker.detect_for_video(img, ts_ms)

        if not res.hand_landmarks:
            return None, None, None

        lm         = res.hand_landmarks[0]
        handedness = res.handedness[0][0].display_name
        states     = self._finger_states(lm, handedness)
        gesture    = next(
            (name for name, pat in self.PATTERNS.items() if states == pat),
            "UNKNOWN",
        )
        self._draw_skeleton(frame, lm)

        palm_ids = [0, 5, 9, 13, 17]
        px = int(np.mean([lm[i].x for i in palm_ids]) * w)
        py = int(np.mean([lm[i].y for i in palm_ids]) * h)
        return gesture, (px, py), (lm[8].x, lm[8].y)

    def close(self):
        self.landmarker.close()


# ── 윈도우 창 제어 ──────────────────────────────────────

class WindowController:
    ACTIONS = {
        "OPEN_PALM": ("최대화",  "maximize"),
        "FIST":      ("최소화",  "minimize"),
        "PEACE":     ("창 복원", "restore"),
    }

    def __init__(self, screen_w, screen_h):
        self.sw = screen_w;  self.sh = screen_h
        self.smooth_x = 0.5; self.smooth_y = 0.5
        self.prev_gest = None; self.dwell_start = None
        self.dwell_ratio = 0.0; self.last_action = 0; self.last_move = 0

    def _run(self, script):
        subprocess.Popen(["osascript", "-e", script],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _move(self, x, y):
        self._run(f"""tell application "System Events"
    try
        set p to first process where frontmost is true
        tell first window of p to set position to {{{x}, {y}}}
    end try
end tell""")

    def _execute(self, action):
        sw, sh = self.sw, self.sh
        scripts = {
            "maximize": f"""tell application "System Events"
    try
        set p to first process where frontmost is true
        tell first window of p
            set position to {{0, 25}}
            set size to {{{sw}, {sh - 25}}}
        end tell
    end try
end tell""",
            "minimize": """tell application "System Events"
    try
        set p to first process where frontmost is true
        set miniaturized of first window of p to true
    end try
end tell""",
            "restore": f"""tell application "System Events"
    try
        set p to first process where frontmost is true
        tell first window of p
            set position to {{{(sw-900)//2}, {(sh-600)//2}}}
            set size to {{900, 600}}
        end tell
    end try
end tell""",
        }
        if action in scripts:
            self._run(scripts[action])

    def update(self, gesture, index_tip_norm):
        now = time.time()
        if gesture == "POINT" and index_tip_norm:
            nx, ny = index_tip_norm
            self.smooth_x += MOVE_SMOOTH * (nx - self.smooth_x)
            self.smooth_y += MOVE_SMOOTH * (ny - self.smooth_y)
            m  = CAM_MARGIN
            mx = max(0.0, min(1.0, (self.smooth_x - m) / max(1 - 2*m, 0.01)))
            my = max(0.0, min(1.0, (self.smooth_y - m) / max(1 - 2*m, 0.01)))
            wx = int(mx * max(self.sw - 500, 0))
            wy = int(my * max(self.sh - 300, 0)) + 25
            if now - self.last_move > MOVE_INTERVAL:
                self.last_move = now
                threading.Thread(target=self._move, args=(wx, wy), daemon=True).start()
            self.prev_gest = gesture; self.dwell_start = None; self.dwell_ratio = 0.0
            return None, 0.0

        if gesture != self.prev_gest:
            self.prev_gest   = gesture
            self.dwell_start = now if (gesture in self.ACTIONS) else None
            self.dwell_ratio = 0.0
            return None, 0.0

        if not gesture or gesture not in self.ACTIONS:
            self.dwell_ratio = 0.0; return None, 0.0
        if now - self.last_action < GESTURE_COOLDOWN:
            return None, 0.0

        elapsed          = now - (self.dwell_start or now)
        self.dwell_ratio = min(elapsed / GESTURE_DWELL, 1.0)

        if elapsed >= GESTURE_DWELL:
            label, action    = self.ACTIONS[gesture]
            self.last_action = now; self.dwell_start = now; self.dwell_ratio = 0.0
            threading.Thread(target=self._execute, args=(action,), daemon=True).start()
            return label, 1.0
        return None, self.dwell_ratio


# ── Claude 대화 ──────────────────────────────────────────

def encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.standard_b64encode(buf).decode("utf-8")


def jarvis_chat(client, user_text: str, frame=None) -> str:
    """다중 턴 대화 — conv_history 유지, 비전 선택적 포함"""
    content = []
    if frame is not None:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": encode_frame(frame),
        }})
    content.append({"type": "text", "text": user_text})

    conv_history.append({
        "role": "user",
        "content": content if len(content) > 1 else user_text,
    })

    # 히스토리 크기 제한
    trimmed = conv_history[-MAX_HISTORY * 2:] if len(conv_history) > MAX_HISTORY * 2 else conv_history

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=trimmed,
        )
        reply = msg.content[0].text
        conv_history.append({"role": "assistant", "content": reply})

        with state["lock"]:
            log = state["conv_log"]
            log.append(("user", user_text))
            log.append(("ai",   reply))
            if len(log) > CONV_DISPLAY * 2:
                state["conv_log"] = log[-CONV_DISPLAY * 2:]

        return reply
    except Exception as e:
        return f"오류: {e}"


def motion_caption(client, frame) -> str:
    """모션 감지용 단문 설명 (Haiku, 빠름)"""
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": encode_frame(frame),
                }},
                {"type": "text",
                 "text": "웹캠에 움직임이 감지됐습니다. 보이는 것을 한 문장으로 한국어로 말해주세요."},
            ]}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"분석 오류: {e}"


def speak(text: str):
    with state["lock"]:
        state["speaking"] = True
    try:
        subprocess.run(["say", "-v", "Yuna", text], timeout=25)
    except Exception:
        subprocess.run(["say", text], timeout=25)
    finally:
        with state["lock"]:
            state["speaking"] = False


def speak_async(text: str):
    threading.Thread(target=speak, args=(text,), daemon=True).start()


# ── 음성 인식 + 웨이크워드 ──────────────────────────────

def _transcribe(audio_np: np.ndarray) -> str:
    """Whisper 로컬 STT — Google FLAC 불필요, M3 네이티브"""
    audio_f32 = audio_np.astype(np.float32) / 32768.0
    result    = _whisper_model.transcribe(
        audio_f32,
        language="ko",
        fp16=False,
        condition_on_previous_text=False,
    )
    text = result.get("text", "").strip()
    if not text:
        raise sr.UnknownValueError()
    return text


def voice_listener(stop_event):
    audio_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    print("[Jarvis] 마이크 대기 중... ('자비스'라고 불러주세요)")
    with sd.InputStream(callback=callback, channels=1,
                        samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, dtype="int16"):
        while not stop_event.is_set():
            speech_chunks = []
            triggered     = False
            silence_start = None

            # 발화 감지
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                if rms > SILENCE_THRESHOLD:
                    triggered = True; silence_start = None
                    speech_chunks.append(chunk)
                elif triggered:
                    speech_chunks.append(chunk)
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        break

            if not triggered or len(speech_chunks) < MIN_SPEECH_CHUNKS:
                continue

            # 자비스가 말하는 중이면 에코 무시
            with state["lock"]:
                if state["speaking"]:
                    continue

            with state["lock"]:
                state["listening"] = True

            audio_np = np.concatenate(speech_chunks).flatten()

            try:
                text = _transcribe(audio_np)
            except sr.UnknownValueError:
                with state["lock"]:
                    state["listening"] = False
                continue
            except Exception as e:
                print(f"[음성 오류] {e}")
                with state["lock"]:
                    state["listening"] = False
                continue

            now_t = time.time()
            text_flat = text.replace(" ", "").lower()

            with state["lock"]:
                currently_awake = now_t < state["awake_until"]

            # ── 슬립 모드 체크 ──────────────────────────
            if currently_awake and any(w in text for w in SLEEP_WORDS):
                with state["lock"]:
                    state["awake"]       = False
                    state["awake_until"] = 0.0
                print(f"[{datetime.now():%H:%M:%S}] → 슬립 모드")
                speak_async("알겠습니다. 필요하시면 불러주세요.")
                with state["lock"]:
                    state["listening"] = False
                continue

            # ── 웨이크워드 체크 ─────────────────────────
            is_wake = any(w in text_flat for w in WAKE_WORDS)

            if not is_wake and not currently_awake:
                # 슬립 상태 + 웨이크워드 없음 → 무시
                with state["lock"]:
                    state["listening"] = False
                continue

            # 웨이크워드 제거 후 명령 추출
            command = text
            if is_wake:
                for w in WAKE_WORDS:
                    command = re.sub(w, "", command, flags=re.IGNORECASE).strip()
                command = command.lstrip(",. ").strip()

                with state["lock"]:
                    state["awake"]       = True
                    state["awake_until"] = now_t + AWAKE_TIMEOUT

                if not command:
                    print(f"[{datetime.now():%H:%M:%S}] 웨이크워드 감지 → 대기 중")
                    speak_async("네, 말씀하세요.")
                    with state["lock"]:
                        state["listening"] = False
                    continue
            else:
                # 이미 깨어있는 상태 → 타이머 갱신
                with state["lock"]:
                    state["awake_until"] = now_t + AWAKE_TIMEOUT

            print(f"[{datetime.now():%H:%M:%S}] 사용자: {command}")
            with state["lock"]:
                state["voice_text"]  = command
                state["new_command"] = True
                state["listening"]   = False


# ── 렌더링 ───────────────────────────────────────────────

def draw_gesture_shape(frame, gesture, dwell_ratio, center):
    if not center or gesture in (None, "UNKNOWN"):
        return
    cx, cy = center
    t = time.time()

    if gesture == "OPEN_PALM":
        r = int(45 + 8 * math.sin(t * 4))
        cv2.circle(frame, (cx, cy), r, (0, 255, 180), 2)
        for i in range(8):
            a = i * math.pi / 4 + t * 0.6
            cv2.line(frame,
                     (int(cx + (r+5)*math.cos(a)),  int(cy + (r+5)*math.sin(a))),
                     (int(cx + (r+20)*math.cos(a)), int(cy + (r+20)*math.sin(a))),
                     (0, 255, 180), 2)

    elif gesture == "FIST":
        r = int(50 - 12 * abs(math.sin(t * 5)))
        cv2.circle(frame, (cx, cy), r,          (30, 80, 255), 2)
        cv2.circle(frame, (cx, cy), max(r-14,5), (20, 50, 200), 1)

    elif gesture == "POINT":
        s = 26
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            tip    = (cx + dx*s, cy + dy*s)
            base_a = (cx + dx*(s//2) - dy*9, cy + dy*(s//2) + dx*9)
            base_b = (cx + dx*(s//2) + dy*9, cy + dy*(s//2) - dx*9)
            cv2.fillPoly(frame, [np.array([tip, base_a, base_b], np.int32)], (255, 200, 50))

    elif gesture == "PEACE":
        s = int(28 + 6 * math.sin(t * 3))
        for dx, dy in [(1,-1),(-1,1),(-1,-1),(1,1)]:
            cv2.arrowedLine(frame, (cx, cy), (cx+dx*s, cy+dy*s),
                            (100, 200, 255), 2, tipLength=0.38)

    if dwell_ratio > 0 and gesture in ("OPEN_PALM", "FIST", "PEACE"):
        color = (0, 255, 255) if dwell_ratio < 0.75 else (50, 255, 50)
        cv2.ellipse(frame, (cx, cy), (62, 62), -90, 0,
                    int(360 * dwell_ratio), color, 3)


def draw_overlay(frame, motion_cnts, motion_conf, gesture, gesture_ko,
                 dwell_ratio, palm_px, action_log, fps, show_help):
    h, w = frame.shape[:2]

    with state["lock"]:
        listening   = state["listening"]
        speaking    = state["speaking"]
        motion_on   = state["motion"]
        awake       = state["awake"]
        awake_until = state["awake_until"]
        conv_log    = list(state["conv_log"])

    # 대화 모드일 때 프레임 테두리 발광 효과
    if awake:
        remaining = max(0, awake_until - time.time())
        pulse     = int(40 + 30 * abs(math.sin(time.time() * 2)))
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, pulse + 180, 255), 3)

    # 제스처 셰이프
    draw_gesture_shape(frame, gesture, dwell_ratio, palm_px)

    # 모션 박스 (손 없을 때)
    if gesture is None:
        for cnt in motion_cnts:
            x, y, bw, bh = cv2.boundingRect(cnt)
            g = int(200 + 55 * motion_conf)
            r = int(255 * (1 - motion_conf))
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, g, r), 2)

    # 상단 바
    cv2.rectangle(frame, (0, 0), (w, 44), (12, 12, 12), -1)
    cv2.putText(frame, f"FPS {fps:.0f}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

    # 대화 모드 타이머 표시
    if awake:
        remaining = max(0, awake_until - time.time())
        cv2.putText(frame, f"● {int(remaining)}s", (w - 65, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)

    if speaking:
        status, color = "JARVIS SPEAKING", (255, 200, 0)
    elif listening:
        status, color = "LISTENING...", (0, 200, 255)
    elif awake:
        status, color = "자비스 대화 중", (0, 220, 255)
    elif gesture and gesture != "UNKNOWN":
        status = gesture_ko + (f"  {int(dwell_ratio*100)}%" if dwell_ratio > 0 else "")
        color  = (50, 255, 180)
    elif motion_on:
        status = f"MOTION {int(motion_conf*100)}%"
        color  = (0, 255, 100)
    else:
        status, color = "웨이크워드: 자비스", (80, 80, 80)

    cv2.putText(frame, status, (w//2 - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    # dwell / motion 바
    if dwell_ratio > 0:
        c = (50, 255, 50) if dwell_ratio >= 1.0 else (0, 200, 255)
        cv2.rectangle(frame, (0, 44), (int(w * dwell_ratio), 47), c, -1)
    elif motion_on and gesture is None:
        cv2.rectangle(frame, (0, 44), (int(w * motion_conf), 47), (0, 255, 100), -1)

    # 액션 플래시
    if action_log:
        tw = cv2.getTextSize(action_log, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
        cv2.putText(frame, action_log, ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 255, 50), 2)

    # ── 대화 로그 패널 ──────────────────────────────────
    if conv_log:
        panel_lines = []
        for role, text in conv_log[-CONV_DISPLAY * 2:]:
            prefix     = "YOU" if role == "user" else " AI"
            clr_role   = (180, 180, 180) if role == "user" else (80, 220, 255)
            max_chars  = 60
            chunks     = [text[i:i+max_chars] for i in range(0, min(len(text), max_chars*2), max_chars)]
            panel_lines.append((prefix, chunks[0] + ("…" if len(chunks) > 1 else ""), clr_role))

        ph = len(panel_lines) * 24 + 14
        py = h - ph - 4
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, py), (w, h), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        for i, (prefix, txt, clr) in enumerate(panel_lines):
            y = py + 18 + i * 24
            cv2.putText(frame, f"{prefix}: {txt}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, clr, 1)

    # 도움말 패널
    if show_help:
        pw  = 300
        ph  = len(HELP_TEXT) * 23 + 18
        px  = w - pw - 10
        pyo = 54
        ov  = frame.copy()
        cv2.rectangle(ov, (px, pyo), (px+pw, pyo+ph), (18, 18, 18), -1)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
        for i, line in enumerate(HELP_TEXT):
            cv2.putText(frame, line, (px+10, pyo+20+i*23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (160, 255, 160) if i == 0 else (170, 170, 170), 1)

    return frame


# ── 메인 ────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client  = None
    if ANTHROPIC_AVAILABLE and api_key:
        client = anthropic.Anthropic(api_key=api_key)
        print("[Jarvis] Claude 대화 모드 활성화")
    else:
        print("[Jarvis] 오프라인 모드 (ANTHROPIC_API_KEY 필요)")

    screen_w, screen_h = get_screen_size()
    print(f"[Jarvis] 스크린 {screen_w}×{screen_h}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 카메라를 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    motion_det = MotionDetector()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    hand_det   = HandGestureDetector(model_path) if (MEDIAPIPE_AVAILABLE and os.path.exists(model_path)) else None
    win_ctrl   = WindowController(screen_w, screen_h)

    stop_event   = threading.Event()
    voice_thread = threading.Thread(target=voice_listener, args=(stop_event,), daemon=True)
    voice_thread.start()

    last_motion_time = 0
    prev_time        = time.time()
    fps              = 0.0
    show_help        = False
    action_log       = ""
    action_log_until = 0.0

    print("[Jarvis] 준비 완료 | '자비스'라고 불러주세요 | h: 도움말 | q: 종료")
    print("-" * 55)
    speak_async("안녕하세요. 자비스입니다. '자비스'라고 불러주시면 대화를 시작할게요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── 손 감지 ─────────────────────────────────────
        gesture = palm_px = index_tip = None
        dwell_ratio = 0.0; gesture_ko = ""

        if hand_det:
            gesture, palm_px, index_tip = hand_det.detect(frame)
            if gesture:
                gesture_ko = HandGestureDetector.GESTURE_KO.get(gesture, gesture)
                triggered_label, dwell_ratio = win_ctrl.update(gesture, index_tip)
                if triggered_label:
                    action_log       = f"{triggered_label} 완료!"
                    action_log_until = now + 1.8
                    print(f"[{now:.1f}] 창 {triggered_label}")

        if now > action_log_until:
            action_log = ""

        # ── 모션 감지 ────────────────────────────────────
        motion_cnts, motion_on, motion_conf = motion_det.update(frame)
        with state["lock"]:
            state["motion"] = motion_on and (gesture is None)
            state["frame"]  = frame.copy()

        # ── 음성 명령 처리 ───────────────────────────────
        with state["lock"]:
            new_cmd    = state["new_command"]
            voice_text = state["voice_text"]

        if new_cmd and client:
            with state["lock"]:
                state["new_command"] = False
                cur_frame = state["frame"].copy()

            # 카메라 영상 포함 여부: 시각 관련 키워드가 있으면 포함
            visual_keywords = ["봐", "봐줘", "뭐야", "뭐가", "있어", "화면", "카메라", "보여"]
            use_vision = any(kw in voice_text for kw in visual_keywords)

            def handle_voice(f, v, vision):
                result = jarvis_chat(client, v, f if vision else None)
                print(f"[{datetime.now():%H:%M:%S}] Jarvis: {result}")
                with state["lock"]:
                    state["analysis"] = result
                speak_async(result)

            threading.Thread(
                target=handle_voice, args=(cur_frame, voice_text, use_vision), daemon=True
            ).start()

        # ── 모션 자동 분석 (슬립 모드만) ─────────────────
        elif (gesture is None and motion_on and client
              and not state["awake"]
              and (now - last_motion_time) > MOTION_COOLDOWN):
            last_motion_time = now
            cur_frame        = frame.copy()

            def handle_motion(f):
                result = motion_caption(client, f)
                print(f"[{datetime.now():%H:%M:%S}] Jarvis(모션): {result}")
                with state["lock"]:
                    state["analysis"] = result
                speak_async(result)

            threading.Thread(target=handle_motion, args=(cur_frame,), daemon=True).start()

        # ── 렌더링 ──────────────────────────────────────
        frame = draw_overlay(
            frame, motion_cnts, motion_conf,
            gesture, gesture_ko, dwell_ratio, palm_px,
            action_log, fps, show_help,
        )
        cv2.imshow("Jarvis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("h"):
            show_help = not show_help

    stop_event.set()
    if hand_det:
        hand_det.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[Jarvis] 종료됨")


if __name__ == "__main__":
    main()
