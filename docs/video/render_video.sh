#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
FPS=30

for svg in "$ROOT"/overlay-*.svg; do
  sips -s format png "$svg" --out "${svg:r}.png" >/dev/null
done

say -v Samantha -r 170 -f "$ROOT/narration.txt" -o "$ROOT/narration.aiff"

still_scene() {
  local input="$1"
  local frames="$2"
  local title="$3"
  local output="$4"
  local overlay="$5"
  local fade_out=$(awk "BEGIN {print ${frames}/${FPS}-0.35}")
  ffmpeg -y -hide_banner -loglevel error -loop 1 -i "$input" -loop 1 -i "$overlay" \
    -filter_complex "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xf4f0e8,zoompan=z='min(zoom+0.00022,1.055)':d=${frames}:s=1920x1080:fps=${FPS}[base];[1:v]format=rgba[ol];[base][ol]overlay=0:0,fade=t=in:st=0:d=0.35,fade=t=out:st=${fade_out}:d=0.35[out]" \
    -map "[out]" \
    -frames:v "$frames" -an -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "$output"
}

# Opening: the same cat image and its fabrication preview.
ffmpeg -y -hide_banner -loglevel error \
  -loop 1 -i "$ROOT/../images/cat3-input.png" \
  -loop 1 -i "$ROOT/../images/cat3-material-preview.png" \
  -loop 1 -i "$ROOT/overlay-01.png" \
  -filter_complex "[0:v]scale=900:820:force_original_aspect_ratio=decrease,pad=960:820:(ow-iw)/2:(oh-ih)/2:0xf4f0e8[left];[1:v]scale=820:820:force_original_aspect_ratio=decrease,pad=960:820:(ow-iw)/2:(oh-ih)/2:0xf4f0e8[right];[left][right]hstack=inputs=2,pad=1920:1080:0:150:0xf4f0e8[base];[2:v]format=rgba[ol];[base][ol]overlay=0:0,fade=t=in:st=0:d=0.5,fade=t=out:st=7.15:d=0.35[out]" \
  -map "[out]" -r 30 -frames:v 225 -an -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "$ROOT/scene-01.mp4"

still_scene "$ROOT/gh-compose.png" 315 "scene 02" "$ROOT/scene-02.mp4" "$ROOT/overlay-02.png"
still_scene "$ROOT/gh-pattern.png" 390 "scene 03" "$ROOT/scene-03.mp4" "$ROOT/overlay-03.png"
still_scene "$ROOT/gh-borrowed.png" 360 "scene 04" "$ROOT/scene-04.mp4" "$ROOT/overlay-04.png"

# Image workflow: show the source and the actual image-field section together.
ffmpeg -y -hide_banner -loglevel error \
  -loop 1 -i "$ROOT/../images/cat3-input.png" -loop 1 -i "$ROOT/gh-image.png" -loop 1 -i "$ROOT/overlay-05.png" \
  -filter_complex "[0:v]scale=720:760:force_original_aspect_ratio=decrease,pad=760:850:(ow-iw)/2:(oh-ih)/2:0xf4f0e8[left];[1:v]crop=760:458:150:0,scale=1120:850:force_original_aspect_ratio=decrease,pad=1160:850:(ow-iw)/2:(oh-ih)/2:0xf4f0e8[right];[left][right]hstack=inputs=2,pad=1920:1080:0:140:0xf4f0e8[base];[2:v]format=rgba[ol];[base][ol]overlay=0:0,fade=t=in:st=0:d=0.35,fade=t=out:st=13.65:d=0.35[out]" \
  -map "[out]" -r 30 -frames:v 420 -an -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "$ROOT/scene-05.mp4"

still_scene "$ROOT/gh-image.png" 360 "scene 06" "$ROOT/scene-06.mp4" "$ROOT/overlay-06.png"
still_scene "$ROOT/../images/cat3-material-preview.png" 390 "scene 07" "$ROOT/scene-07.mp4" "$ROOT/overlay-07.png"
still_scene "$ROOT/gh-export.png" 300 "scene 08" "$ROOT/scene-08.mp4" "$ROOT/overlay-08.png"

ffmpeg -y -hide_banner -loglevel error \
  -i "$ROOT/scene-01.mp4" -i "$ROOT/scene-02.mp4" -i "$ROOT/scene-03.mp4" -i "$ROOT/scene-04.mp4" \
  -i "$ROOT/scene-05.mp4" -i "$ROOT/scene-06.mp4" -i "$ROOT/scene-07.mp4" -i "$ROOT/scene-08.mp4" \
  -i "$ROOT/narration.aiff" \
  -filter_complex "[0:v]setsar=1[v0];[1:v]setsar=1[v1];[2:v]setsar=1[v2];[3:v]setsar=1[v3];[4:v]setsar=1[v4];[5:v]setsar=1[v5];[6:v]setsar=1[v6];[7:v]setsar=1[v7];[v0][v1][v2][v3][v4][v5][v6][v7]concat=n=8:v=1:a=0[video];[8:a]adelay=500|500,apad=pad_dur=2,volume=1.05[audio]" \
  -map "[video]" -map "[audio]" -t 92 \
  -c:v libx264 -preset medium -crf 19 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart \
  "$ROOT/JIEJING_Demo_90s.mp4"

ffmpeg -y -hide_banner -loglevel error -ss 2 -i "$ROOT/JIEJING_Demo_90s.mp4" -frames:v 1 "$ROOT/JIEJING_Demo_Cover.png"
ffmpeg -y -hide_banner -loglevel error -ss 0 -t 12 -i "$ROOT/JIEJING_Demo_90s.mp4" \
  -vf "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
  -loop 0 "$ROOT/JIEJING_Demo_Preview.gif"

