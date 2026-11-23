#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
SYNC="$ROOT/sync"
FPS=30
mkdir -p "$SYNC"

for index in {1..8}; do
  printf -v n "%02d" "$index"
  say -v Samantha -r 170 -f "$ROOT/cues/$n.txt" -o "$SYNC/cue-$n.aiff"
  sips -s format png "$ROOT/overlay-$n.svg" --out "$SYNC/overlay-$n.png" >/dev/null
  sips -s format png "$ROOT/caption-$n.svg" --out "$SYNC/caption-$n.png" >/dev/null
done

render_still() {
  local n="$1"
  local input="$2"
  local audio="$SYNC/cue-$n.aiff"
  local duration=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$audio")
  local frames=$(awk "BEGIN {print int((${duration}+0.04)*${FPS}+0.999)}")
  local fade_out=$(awk "BEGIN {v=${duration}-0.25; if(v<0)v=0; print v}")
  ffmpeg -y -hide_banner -loglevel error \
    -loop 1 -i "$input" -loop 1 -i "$SYNC/overlay-$n.png" -loop 1 -i "$SYNC/caption-$n.png" -i "$audio" \
    -filter_complex "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xf4f0e8,zoompan=z='min(zoom+0.00022,1.045)':d=${frames}:s=1920x1080:fps=${FPS}[base];[1:v]format=rgba[top];[2:v]format=rgba[cap];[base][top]overlay=0:0[mid];[mid][cap]overlay=0:0,fade=t=in:st=0:d=0.25,fade=t=out:st=${fade_out}:d=0.25[out]" \
    -map "[out]" -map 3:a -frames:v "$frames" -shortest \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k "$SYNC/scene-$n.mp4"
}

render_split() {
  local n="$1"
  local left="$2"
  local right="$3"
  local mode="$4"
  local audio="$SYNC/cue-$n.aiff"
  local duration=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$audio")
  local frames=$(awk "BEGIN {print int((${duration}+0.04)*${FPS}+0.999)}")
  local fade_out=$(awk "BEGIN {v=${duration}-0.25; if(v<0)v=0; print v}")
  if [[ "$mode" == "opening" ]]; then
    local layout="[0:v]scale=900:800:force_original_aspect_ratio=decrease,pad=960:820:(ow-iw)/2:(oh-ih)/2:0xf4f0e8[left];[1:v]scale=820:800:force_original_aspect_ratio=decrease,pad=960:820:(ow-iw)/2:(oh-ih)/2:0xf4f0e8[right];[left][right]hstack=inputs=2,pad=1920:1080:0:150:0xf4f0e8[base]"
  else
    local layout="[0:v]scale=720:760:force_original_aspect_ratio=decrease,pad=760:850:(ow-iw)/2:(oh-ih)/2:0xf4f0e8[left];[1:v]crop=760:458:150:0,scale=1120:850:force_original_aspect_ratio=decrease,pad=1160:850:(ow-iw)/2:(oh-ih)/2:0xf4f0e8[right];[left][right]hstack=inputs=2,pad=1920:1080:0:140:0xf4f0e8[base]"
  fi
  ffmpeg -y -hide_banner -loglevel error \
    -loop 1 -i "$left" -loop 1 -i "$right" -loop 1 -i "$SYNC/overlay-$n.png" -loop 1 -i "$SYNC/caption-$n.png" -i "$audio" \
    -filter_complex "${layout};[2:v]format=rgba[top];[3:v]format=rgba[cap];[base][top]overlay=0:0[mid];[mid][cap]overlay=0:0,fade=t=in:st=0:d=0.25,fade=t=out:st=${fade_out}:d=0.25[out]" \
    -map "[out]" -map 4:a -r 30 -frames:v "$frames" -shortest \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k "$SYNC/scene-$n.mp4"
}

render_split 01 "$ROOT/../images/cat3-input.png" "$ROOT/../images/cat3-material-preview.png" opening
render_still 02 "$ROOT/gh-compose.png"
render_still 03 "$ROOT/gh-pattern.png"
render_still 04 "$ROOT/gh-borrowed.png"
render_split 05 "$ROOT/../images/cat3-input.png" "$ROOT/gh-image.png" image
render_still 06 "$ROOT/gh-image.png"
render_still 07 "$ROOT/../images/cat3-material-preview.png"
render_still 08 "$ROOT/gh-export.png"

ffmpeg -y -hide_banner -loglevel error \
  -i "$SYNC/scene-01.mp4" -i "$SYNC/scene-02.mp4" -i "$SYNC/scene-03.mp4" -i "$SYNC/scene-04.mp4" \
  -i "$SYNC/scene-05.mp4" -i "$SYNC/scene-06.mp4" -i "$SYNC/scene-07.mp4" -i "$SYNC/scene-08.mp4" \
  -filter_complex "[0:v]setsar=1[v0];[1:v]setsar=1[v1];[2:v]setsar=1[v2];[3:v]setsar=1[v3];[4:v]setsar=1[v4];[5:v]setsar=1[v5];[6:v]setsar=1[v6];[7:v]setsar=1[v7];[v0][0:a][v1][1:a][v2][2:a][v3][3:a][v4][4:a][v5][5:a][v6][6:a][v7][7:a]concat=n=8:v=1:a=1[video][audio]" \
  -map "[video]" -map "[audio]" -c:v libx264 -preset medium -crf 19 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart \
  "$ROOT/JIEJING_Demo_90s.mp4"

python3 "$ROOT/make_synced_srt.py"
ffmpeg -y -hide_banner -loglevel error -ss 2 -i "$ROOT/JIEJING_Demo_90s.mp4" -frames:v 1 "$ROOT/JIEJING_Demo_Cover.png"
ffmpeg -y -hide_banner -loglevel error -ss 0 -t 12 -i "$ROOT/JIEJING_Demo_90s.mp4" \
  -vf "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
  -loop 0 "$ROOT/JIEJING_Demo_Preview.gif"

