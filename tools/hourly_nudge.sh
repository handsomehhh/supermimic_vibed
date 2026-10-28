#!/usr/bin/env bash
# Hourly nudge from outside Claude Code (installed in the user crontab): sends the progress prompt
# to the session every hour; skips only while a turn is in progress (transcript written very recently).
SESSION=05205e29-6c49-4244-b904-9bdaeac3c18b
TRANSCRIPT=/home/lxz/.claude/projects/-home-lxz-supermimic-vibed/$SESSION.jsonl
IDLE_MINUTES=3
LOG=/home/lxz/supermimic_vibed/logs/hourly_nudge.log
PROMPT="报告当前进度，如果发生停滞，请你继续本轮任务"
export PATH=/home/lxz/.nvm/versions/node/v24.14.0/bin:/usr/local/bin:/usr/bin:/bin
# cron has no proxy settings; without them the API returns 403
[ -f /home/lxz/.config/hourly_nudge.env ] && . /home/lxz/.config/hourly_nudge.env

mkdir -p "$(dirname "$LOG")"
exec 9>/tmp/hourly_nudge.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) previous nudge still running" >> "$LOG"; exit 0; }
idle=$(( ($(date +%s) - $(stat -c %Y "$TRANSCRIPT")) / 60 ))
if [ "$idle" -lt "$IDLE_MINUTES" ]; then
  echo "$(date -u +%FT%TZ) turn in progress (idle ${idle} min), skip" >> "$LOG"
  exit 0
fi
echo "$(date -u +%FT%TZ) sending prompt (idle ${idle} min)" >> "$LOG"
cd /home/lxz/supermimic_vibed
timeout 3h claude -p --resume "$SESSION" --dangerously-skip-permissions "$PROMPT" < /dev/null >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) nudge finished (exit $?)" >> "$LOG"
