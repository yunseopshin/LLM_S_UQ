#!/bin/bash
# Claude Code에 Phase 파일을 pipe로 주입하여 실행
# 사용법:
#   ./run_phase.sh 0_init
#   ./run_phase.sh 1_1_generation

set -e

PHASE=$1
FILE="prompts/phase_${PHASE}.md"

if [ ! -f "$FILE" ]; then
    echo "Error: $FILE not found"
    echo "Available phases:"
    ls prompts/ | grep -oP 'phase_\K[^.]+' | sort
    exit 1
fi

echo "=== Running Phase: $PHASE ==="
echo "File: $FILE"
echo ""

# Claude Code CLI 호출
# (실제 설치된 command 이름에 따라 'claude' 또는 'claude-code' 등으로 수정)
cat "$FILE" | claude
