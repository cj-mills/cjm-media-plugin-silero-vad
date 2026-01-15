import sys
import os
import json
import time
from pathlib import Path

# Add paths to find local libs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cjm_media_plugin_silero_vad.plugin import SileroVADPlugin

def title(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def run_vad_test():
    title("TEST: Silero VAD Plugin (Analysis & Caching)")

    # 1. Setup
    # Update this path to your actual test file
    audio_file = "/mnt/SN850X_8TB_EXT4/Projects/GitHub/cj-mills/cjm-transcription-plugin-voxtral-hf/test_files/02 - 1. Laying Plans.mp3"
    
    if not os.path.exists(audio_file):
        print(f"ERROR: Test file not found at {audio_file}")
        return

    # 2. Initialize Plugin
    print("Initializing Plugin...")
    plugin = SileroVADPlugin()
    
    # We point to a temp dir for the DB to avoid polluting your main dev setup if desired,
    # or let it use the default env path. Let's use default to test persistence.
    plugin.initialize({
        "threshold": 0.5,
        "min_speech_duration_ms": 250
    })

    # 3. First Run (Processing)
    print("\n--- Run 1: Processing Audio (Cold Start) ---")
    start_time = time.time()
    
    # Note: We pass force=True to ensure we actually test the inference logic first
    result_1 = plugin.execute(audio_file, force=True)
    
    duration_1 = time.time() - start_time
    print(f"Result 1 received in {duration_1:.2f}s")
    
    # Validation
    assert result_1 is not None
    assert len(result_1.ranges) > 0
    print(f"  -> Detected {len(result_1.ranges)} speech segments.")
    print(f"  -> Total Speech: {result_1.metadata['total_speech']:.2f}s")
    
    # Print first 3 segments
    print("  -> Sample Segments:")
    for i, r in enumerate(result_1.ranges[:3]):
        print(f"     [{i}] {r.start:.2f}s - {r.end:.2f}s (conf: {r.confidence})")

    # 4. Second Run (Caching)
    print("\n--- Run 2: Checking Cache (Hot Start) ---")
    start_time = time.time()
    
    # force=False should hit the DB
    result_2 = plugin.execute(audio_file, force=False)
    
    duration_2 = time.time() - start_time
    print(f"Result 2 received in {duration_2:.2f}s")
    
    # Validation
    assert len(result_2.ranges) == len(result_1.ranges)
    assert result_2.ranges[0].start == result_1.ranges[0].start
    
    # Performance check (Cache should be instantaneous)
    if duration_2 < 0.1: # SQLite lookup is usually < 10ms
        print("  -> PASS: Cache hit successful (sub-100ms response).")
    else:
        print(f"  -> WARNING: Response took {duration_2:.2f}s. Cache might have missed?")

    # 5. Cleanup
    plugin.cleanup()
    print("\n[SUCCESS] VAD Plugin verified.")

if __name__ == "__main__":
    try:
        run_vad_test()
    except Exception as e:
        print(f"\n!!! FAILED !!!\n{e}")
        import traceback
        traceback.print_exc()