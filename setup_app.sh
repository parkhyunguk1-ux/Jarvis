#!/bin/bash
# setup_app.sh — Jarvis.app + Siri Shortcut 생성 스크립트

set -e
JARVIS_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$HOME/Applications/Jarvis.app"
CONTENTS="$APP/Contents"
PYTHON3="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3"

echo "══════════════════════════════════════════"
echo "  Jarvis.app 설치"
echo "══════════════════════════════════════════"

# ── 1. 앱 디렉토리 생성 ────────────────────────────────
mkdir -p "$CONTENTS/MacOS"
mkdir -p "$CONTENTS/Resources"

# ── 2. 런처 실행파일 ────────────────────────────────────
cat > "$CONTENTS/MacOS/Jarvis" << LAUNCHER
#!/bin/bash
JARVIS_DIR="$JARVIS_DIR"
LOGFILE="\$JARVIS_DIR/jarvis.log"

# 중복 실행 방지
if pgrep -f "python3.*jarvis.py" > /dev/null 2>&1; then
    osascript -e 'display notification "자비스가 이미 실행 중입니다" with title "Jarvis"'
    exit 0
fi

# .env 에서 API 키 로드
[ -f "\$JARVIS_DIR/.env" ] && export \$(grep -v '^\#' "\$JARVIS_DIR/.env" | xargs) 2>/dev/null || true

# MediaPipe 모델 자동 다운로드
MODEL="\$JARVIS_DIR/hand_landmarker.task"
if [ ! -f "\$MODEL" ]; then
    osascript -e 'display notification "손 인식 모델을 다운로드 중..." with title "Jarvis"'
    curl -sL "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -o "\$MODEL" 2>/dev/null || true
fi

osascript -e 'display notification "AI 비서가 시작됩니다 — '\''자비스'\''라고 불러주세요" with title "Jarvis" subtitle "카메라 창을 확인하세요"'

cd "\$JARVIS_DIR"
exec "$PYTHON3" "\$JARVIS_DIR/jarvis.py" >> "\$LOGFILE" 2>&1
LAUNCHER
chmod +x "$CONTENTS/MacOS/Jarvis"

# ── 3. Info.plist ──────────────────────────────────────
cat > "$CONTENTS/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>           <string>Jarvis</string>
    <key>CFBundleDisplayName</key>    <string>Jarvis</string>
    <key>CFBundleIdentifier</key>     <string>com.parkhyunguk.jarvis</string>
    <key>CFBundleExecutable</key>     <string>Jarvis</string>
    <key>CFBundleIconFile</key>       <string>AppIcon</string>
    <key>CFBundleVersion</key>        <string>1.0</string>
    <key>CFBundleShortVersionString</key> <string>1.0</string>
    <key>CFBundlePackageType</key>    <string>APPL</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSMinimumSystemVersion</key> <string>12.0</string>
    <key>NSCameraUsageDescription</key>
        <string>모션 감지 및 AI 분석을 위해 카메라를 사용합니다.</string>
    <key>NSMicrophoneUsageDescription</key>
        <string>음성 명령 인식을 위해 마이크를 사용합니다.</string>
</dict>
</plist>
PLIST

# ── 4. 앱 아이콘 생성 (Python) ─────────────────────────
echo "🎨 아이콘 생성 중..."
"$PYTHON3" << 'ICONPY'
import os, math, struct, zlib

def make_png(size):
    cx = cy = size // 2
    R = int(size * 0.46)
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy)
            if d > R:
                row += bytes([15, 15, 30])
                continue
            t = d / R
            # 다크 네이비 → 미드나잇 블루 그라디언트
            br = int(10 + 30 * t)
            bg = int(20 + 60 * t)
            bb = int(80 + 120 * t)

            # 'J' 레터 그리기
            lw = max(2, int(size * 0.055))   # 선 굵기
            lx = cx + int(size * 0.07)        # J 세로선 x
            ty = cy - int(size * 0.24)        # 상단 y
            by = cy + int(size * 0.18)        # 하단 y
            on = False
            # 세로 획
            if abs(x - lx) < lw and ty <= y <= by:
                on = True
            # 상단 가로 획
            if abs(y - ty) < lw and lx - lw * 4 <= x <= lx + lw:
                on = True
            # 하단 곡선
            if y >= by - lw:
                ccx, ccy = lx - lw * 3, by
                cd = math.hypot(x - ccx, y - ccy)
                if lw * 1.2 < cd < lw * 3.8 and x < lx:
                    on = True

            if on:
                row += bytes([230, 240, 255])
            else:
                # 부드러운 엣지 안티앨리어싱 (테두리)
                if R - d < 2:
                    f = (R - d) / 2
                    row += bytes([int(br*f), int(bg*f), int(bb*f)])
                else:
                    row += bytes([br, bg, bb])
        rows.append(bytes([0]) + bytes(row))

    raw = zlib.compress(b''.join(rows), 9)

    def chunk(tag, data):
        c = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', c)

    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', raw)
            + chunk(b'IEND', b''))

iconset = os.path.expanduser(
    "~/Applications/Jarvis.app/Contents/Resources/AppIcon.iconset")
os.makedirs(iconset, exist_ok=True)

for sz in [16, 32, 64, 128, 256, 512]:
    data = make_png(sz)
    open(f"{iconset}/icon_{sz}x{sz}.png", 'wb').write(data)
    open(f"{iconset}/icon_{sz}x{sz}@2x.png", 'wb').write(make_png(sz * 2))

import subprocess
icns = os.path.expanduser(
    "~/Applications/Jarvis.app/Contents/Resources/AppIcon.icns")
subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)

import shutil; shutil.rmtree(iconset)
print("  아이콘 OK")
ICONPY

# ── 5. Launch Services 등록 ───────────────────────────
/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister \
    -f "$APP" 2>/dev/null || true

# ── 6. Siri Shortcut 생성 ─────────────────────────────
echo "🎙️  Siri Shortcut 생성 중..."

SHORTCUT_NAME="자비스"
SHORTCUT_DIR="$HOME/Library/Shortcuts"
mkdir -p "$SHORTCUT_DIR"

# Shortcut plist (XML) 생성
cat > "/tmp/jarvis_shortcut.plist" << SPLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>WFWorkflowActions</key>
  <array>
    <dict>
      <key>WFWorkflowActionIdentifier</key>
      <string>is.workflow.actions.openapp</string>
      <key>WFWorkflowActionParameters</key>
      <dict>
        <key>WFAppIdentifier</key>
        <string>com.parkhyunguk.jarvis</string>
      </dict>
    </dict>
  </array>
  <key>WFWorkflowClientVersion</key>
  <string>2600</string>
  <key>WFWorkflowMinimumClientVersion</key>
  <integer>900</integer>
  <key>WFWorkflowName</key>
  <string>자비스</string>
  <key>WFWorkflowIcon</key>
  <dict>
    <key>WFWorkflowIconStartColor</key>
    <integer>431817727</integer>
    <key>WFWorkflowIconGlyphNumber</key>
    <integer>59511</integer>
  </dict>
</dict>
</plist>
SPLIST

# plist → binary → .shortcut 파일로 가져오기
plutil -convert binary1 "/tmp/jarvis_shortcut.plist" -o "/tmp/jarvis.shortcut" 2>/dev/null && \
shortcuts import "/tmp/jarvis.shortcut" --name "자비스" 2>/dev/null && \
echo "  Shortcut '자비스' OK" || \
echo "  ⚠️  Shortcut 수동 추가 필요 (아래 안내 참조)"

rm -f /tmp/jarvis_shortcut.plist /tmp/jarvis.shortcut

# ── 완료 ──────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  ✅ 설치 완료!"
echo "══════════════════════════════════════════"
echo ""
echo "  📂 앱 위치: ~/Applications/Jarvis.app"
echo ""
echo "  🚀 실행 방법:"
echo "    • Finder → 응용 프로그램 → Jarvis 더블클릭"
echo "    • Spotlight: Cmd+Space → 'Jarvis' 검색"
echo "    • Siri: 'Hey Siri, 자비스 켜줘'"
echo ""
echo "  🎙️  Siri에 '자비스' 단축어를 Siri에 추가하려면:"
echo "    Shortcuts.app → '자비스' 단축어 → ⚙️ → Siri에 추가"
echo ""

# Dock에 추가 여부 묻기
read -p "Dock에 Jarvis 추가할까요? (y/n): " ADD_DOCK
if [[ "$ADD_DOCK" == "y" || "$ADD_DOCK" == "Y" ]]; then
    defaults write com.apple.dock persistent-apps -array-add \
        "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>$APP</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>" \
        2>/dev/null
    killall Dock 2>/dev/null
    echo "  ✅ Dock에 추가됨"
fi

echo ""
echo "  지금 바로 실행하려면:"
echo "  open ~/Applications/Jarvis.app"
