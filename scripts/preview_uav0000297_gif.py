#!/usr/bin/env python3
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

MP4 = Path("assets/demo_outputs/uav0000297_00000_v_ds.mp4")
OUT_GIF = Path("assets/demo_outputs/uav0000297_00000_v_ds.gif")

def detect_crop_bounds(frames, threshold=8):
    h, w = frames[0].shape[:2]
    crop_top = 0
    crop_bottom = h
    for frame in frames:
        gray = frame.mean(axis=2)
        top = 0
        for i in range(h):
            if gray[i].mean() > threshold:
                top = i
                break
        bottom = h
        for i in range(h - 1, -1, -1):
            if gray[i].mean() > threshold:
                bottom = i + 1
                break
        crop_top = max(crop_top, top)
        crop_bottom = min(crop_bottom, bottom)
    if crop_bottom <= crop_top:
        return 0, h
    return crop_top, crop_bottom

def main():
    if not MP4.exists():
        raise SystemExit(f"Missing {MP4}")

    reader = imageio.get_reader(str(MP4))
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30)

    sample_idxs = [0, min(10, int(fps)), min(20, int(2 * fps)), min(30, int(3 * fps))]
    samples = []
    for idx in sample_idxs:
        try:
            samples.append(reader.get_data(idx))
        except Exception:
            break

    if not samples:
        reader.close()
        raise SystemExit("No frames available for crop detection.")

    crop_top, crop_bottom = detect_crop_bounds(samples)
    reader.close()

    reader = imageio.get_reader(str(MP4))
    step = max(1, int(round(fps / 8)))
    max_frames = int(4 * fps)

    out_frames = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        if i % step != 0:
            continue
        frame = frame[crop_top:crop_bottom]
        out_frames.append(frame)

    reader.close()

    if not out_frames:
        raise SystemExit("No frames captured for GIF.")

    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(OUT_GIF), out_frames, duration=1 / 8)
    print(f"Wrote {OUT_GIF} (crop_top={crop_top}, crop_bottom={crop_bottom})")

if __name__ == "__main__":
    main()
