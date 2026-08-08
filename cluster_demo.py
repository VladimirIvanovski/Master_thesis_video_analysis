"""
cluster_demo.py — Run 24 CPU workers across both nodes
Open Ray Dashboard at http://192.168.1.55:8265 to watch utilization
"""
import ray, time, os, glob

ray.init(address="auto")

print(f"Cluster resources: {ray.cluster_resources()}")

# ── find up to 60 audio files to transcribe ───────────────────────────────
audio_files = glob.glob("results_4/**/*.wav", recursive=True)[:200]
if not audio_files:
    audio_files = glob.glob("results_4/**/*.mp3", recursive=True)[:200]
# repeat files to have enough work if fewer than 200
while len(audio_files) < 200:
    audio_files = (audio_files * 4)[:200]

print(f"Found {len(audio_files)} audio files")

@ray.remote(num_cpus=1)
def cpu_transcribe(audio_path):
    import wave, struct, math
    # CPU-intensive pure Python work: read audio + compute RMS energy per chunk
    try:
        with wave.open(audio_path, 'rb') as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(min(n_frames, 44100 * 10))  # up to 10s
        samples = struct.unpack(f"{len(raw)//2}h", raw[:len(raw) - len(raw)%2])
        chunk = 512
        rms_values = []
        # run 5 passes to keep CPU busy longer
        for _ in range(5):
            for i in range(0, len(samples) - chunk, chunk):
                s = samples[i:i+chunk]
                rms = math.sqrt(sum(x*x for x in s) / chunk)
                rms_values.append(rms)
        avg_rms = sum(rms_values) / max(len(rms_values), 1)
    except Exception:
        avg_rms = 0.0
    return {"file": os.path.basename(audio_path), "avg_rms": round(avg_rms, 2)}

print(f"\nDispatching {len(audio_files)} tasks across 24 CPU workers...")
print("Watch http://192.168.1.55:8265 — you should see CPU usage on both nodes!\n")

t0 = time.time()
refs = [cpu_transcribe.remote(f) for f in audio_files]

done = 0
while refs:
    ready, refs = ray.wait(refs, num_returns=min(4, len(refs)), timeout=5)
    results = ray.get(ready)
    done += len(results)
    elapsed = time.time() - t0
    print(f"  {done}/{len(audio_files)+done} done | {elapsed:.1f}s elapsed | "
          f"{done/elapsed:.2f} files/s")

total = time.time() - t0
print(f"\nDone! {len(audio_files)} files in {total:.1f}s "
      f"({len(audio_files)/total:.2f} files/s)")
print("Check the dashboard screenshot now!")
