#!/bin/bash
# start_jarvis.sh — Siri Shortcut에서 호출하는 자비스 런처
# 사용: "Hey Siri, 자비스 시작해" → Shortcuts.app에서 이 스크립트 실행

DIR="$(cd "$(dirname "$0")" && pwd)"

# .env 파일에서 API 키 로드 (있을 경우)
if [ -f "$DIR/.env" ]; then
    export $(grep -v '^#' "$DIR/.env" | xargs) 2>/dev/null
fi

# 이미 실행 중이면 창 포커스만
if pgrep -f "jarvis.py" > /dev/null; then
    osascript -e 'tell application "Terminal" to activate' 2>/dev/null
    exit 0
fi

# 새 Terminal 창에서 실행
osascript <<EOF
tell application "Terminal"
    activate
    set newTab to do script "cd '$DIR' && python3 jarvis.py"
end tell
EOF
