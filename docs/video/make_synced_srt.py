from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def stamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


cursor = 0.0
blocks = []
for index in range(1, 9):
    cue = ROOT / "cues" / f"{index:02d}.txt"
    scene = ROOT / "sync" / f"scene-{index:02d}.mp4"
    text = cue.read_text(encoding="utf-8").strip()
    end = cursor + duration(scene)
    blocks.append(f"{index}\n{stamp(cursor)} --> {stamp(end)}\n{text}\n")
    cursor = end

(ROOT / "JIEJING_Demo_90s.srt").write_text("\n".join(blocks), encoding="utf-8")
print(f"subtitle_duration={cursor:.3f}")

