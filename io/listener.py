import sounddevice
import numpy as np
import queue
from pathlib import Path

import yaml

audio_queue = queue.Queue()

config_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def list_devices():
    devices = sounddevice.query_devices()
    for idx, device in enumerate(devices):
        print(f"{idx}: {device['name']} - {device['max_input_channels']} input channels,\
               {device['max_output_channels']} output channels")


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio callback status: {status}")
    print(f"Received audio chunk with shape: {indata.shape}")
    audio_queue.put(indata.copy())
