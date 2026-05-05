"""
Jarvis - 모션 인식 + 음성 인식 AI 어시스턴트
- 음성 명령 → Claude Vision 분석 → 음성 응답
- 모션 감지 시 자동 분석
- q 키: 종료
"""

import cv2
import time
import os
import base64
import queue
import threading
import subprocess
import numpy as np
import speech_recognition as sr
import sounddevice as sd
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── 설정 ──────────────────────────────────────────────
MOTION_THRESHOLD   = 1500   # 모션 감지 픽셀 면적
MOTION_COOLDOWN    = 10     # 모션 자동 분석 간격 (초)
SAMPLE_RATE        = 16000  # 마이크 샘플레이트
CHUNK_SIZE         = 1024   # 오디오 청크 크기
SILENCE_THRESHOLD  = 600    # 음성 감지 RMS 임계값
SILENCE_DURATION   = 1.2    # 침묵 지속 시간(초) → 발화 종료 판단
MIN_SPEECH_CHUNKS  = 8      # 최소 음성 청크 수 (너무 짧으면 무시)
BLUR_SIZE          = 21
# ──────────────────────────────────────────────────────

# 공유 상태
state = {
    "lock": threading.Lock(),
    "frame": None,
    "analysis": "",
    "voice_text": "",
    "listening": False,
    "speaking": False,
    "motion": False,
    "new_command": False,
}


def encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.standard_b64encode(buf).decode("utf-8")


def ask_claude(client, frame, user_text: str = "") -> str:
    prompt = (
        f"사용자가 '{user_text}'라고 말했습니다. "
        "이 웹캠 영상에서 보이는 것을 바탕으로 한국어 두 문장 이내로 답해주세요."
        if user_text else
        "이 웹캠 영상에서 움직임이 감지됐습니다. 무엇이 보이는지 한 문장으로 말해주세요. 한국어로 답하세요."
    )
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encode_frame(frame),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return msg.content[0].text
    except Exception as e:
        return f"분석 오류: {e}"


def speak(text: str):
    """macOS TTS - Yuna(한국어) 음성"""
    with state["lock"]:
        state["speaking"] = True
    try:
        subprocess.run(["say", "-v", "Yuna", text], timeout=20)
    except Exception:
        subprocess.run(["say", text], timeout=20)
    finally:
        with state["lock"]:
            state["speaking"] = False


def speak_async(text: str):
    threading.Thread(target=speak, args=(text,), daemon=True).start()


def voice_listener(stop_event):
    """백그라운드 음성 인식 스레드"""
    recognizer = sr.Recognizer()
    audio_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    print("[Jarvis] 마이크 대기 중...")
    with sd.InputStream(
        callback=callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype="int16",
    ):
        while not stop_event.is_set():
            speech_chunks = []
            triggered = False
            silence_start = None

            # 발화 감지 루프
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

                if rms > SILENCE_THRESHOLD:
                    triggered = True
                    silence_start = None
                    speech_chunks.append(chunk)
                elif triggered:
                    speech_chunks.append(chunk)
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        break  # 침묵 → 발화 끝

            if not triggered or len(speech_chunks) < MIN_SPEECH_CHUNKS:
                continue

            # Jarvis가 말하는 중이면 에코 방지
            with state["lock"]:
                if state["speaking"]:
                    continue

            with state["lock"]:
                state["listening"] = True

            audio_np = np.concatenate(speech_chunks).flatten()
            audio_data = sr.AudioData(audio_np.tobytes(), SAMPLE_RATE, 2)

            try:
                text = recognizer.recognize_google(audio_data, language="ko-KR")
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] 사용자: {text}")
                with state["lock"]:
                    state["voice_text"] = text
                    state["new_command"] = True
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"[음성 인식 오류] {e}")
            finally:
                with state["lock"]:
                    state["listening"] = False


def draw_overlay(frame, contours, fps):
    h, w = frame.shape[:2]

    with state["lock"]:
        motion      = state["motion"]
        analysis    = state["analysis"]
        listening   = state["listening"]
        speaking    = state["speaking"]
        voice_text  = state["voice_text"]

    # 모션 박스
    for cnt in contours:
        if cv2.contourArea(cnt) < MOTION_THRESHOLD:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 100), 2)

    # 상단 상태 바
    cv2.rectangle(frame, (0, 0), (w, 40), (15, 15, 15), -1)
    cv2.putText(frame, f"FPS {fps:.0f}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    if speaking:
        status, color = "JARVIS SPEAKING", (255, 200, 0)
    elif listening:
        status, color = "LISTENING...", (0, 200, 255)
    elif motion:
        status, color = "MOTION DETECTED", (0, 255, 100)
    else:
        status, color = "MONITORING", (100, 100, 100)

    cv2.putText(frame, status, (w // 2 - 90, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # 하단: 사용자 발화 + 분석 결과
    if voice_text or analysis:
        cv2.rectangle(frame, (0, h - 80), (w, h), (15, 15, 15), -1)
        if voice_text:
            vt = voice_text[:65] + ("..." if len(voice_text) > 65 else "")
            cv2.putText(frame, f"YOU: {vt}", (8, h - 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        if analysis:
            an = analysis[:68] + ("..." if len(analysis) > 68 else "")
            cv2.putText(frame, f"AI : {an}", (8, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 220, 255), 1)

    return frame


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = None
    if ANTHROPIC_AVAILABLE and api_key:
        client = anthropic.Anthropic(api_key=api_key)
        print("[Jarvis] Claude Vision 활성화됨")
    else:
        print("[Jarvis] 모션 감지 전용 모드 (ANTHROPIC_API_KEY 없음)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 카메라를 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    stop_event = threading.Event()
    voice_thread = threading.Thread(target=voice_listener, args=(stop_event,), daemon=True)
    voice_thread.start()

    last_motion_time = 0
    prev_time = time.time()
    fps = 0.0

    print("[Jarvis] 준비 완료 | 말을 걸거나 움직여 보세요 | q: 종료")
    print("-" * 50)
    speak_async("안녕하세요. 자비스입니다. 무엇을 도와드릴까요?")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

        # 모션 감지
        blur = cv2.GaussianBlur(frame, (BLUR_SIZE, BLUR_SIZE), 0)
        fg   = bg_sub.apply(blur)
        fg   = cv2.dilate(fg, kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large = [c for c in contours if cv2.contourArea(c) >= MOTION_THRESHOLD]
        motion_now = len(large) > 0

        with state["lock"]:
            state["motion"] = motion_now
            state["frame"] = frame.copy()

        # ① 음성 명령 처리
        with state["lock"]:
            new_cmd = state["new_command"]
            voice_text = state["voice_text"]

        if new_cmd and client:
            with state["lock"]:
                state["new_command"] = False
                cur_frame = state["frame"].copy()

            def handle_voice(f, v):
                result = ask_claude(client, f, v)
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] Jarvis: {result}")
                with state["lock"]:
                    state["analysis"] = result
                speak_async(result)

            threading.Thread(target=handle_voice, args=(cur_frame, voice_text), daemon=True).start()

        # ② 모션 자동 분석 (쿨다운)
        elif motion_now and client and (now - last_motion_time) > MOTION_COOLDOWN:
            last_motion_time = now
            cur_frame = frame.copy()

            def handle_motion(f):
                result = ask_claude(client, f)
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] Jarvis(모션): {result}")
                with state["lock"]:
                    state["analysis"] = result
                speak_async(result)

            threading.Thread(target=handle_motion, args=(cur_frame,), daemon=True).start()

        frame = draw_overlay(frame, large, fps)
        cv2.imshow("Jarvis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    print("[Jarvis] 종료됨")


if __name__ == "__main__":
    main()
