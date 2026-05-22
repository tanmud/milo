from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncGenerator

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel


SUPPORTED_VAD_FRAME_MS = {10, 20, 30}


def list_devices() -> None:
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device["max_input_channels"] <= 0:
            continue
        print(
            f"{index}: {device['name']} - {device['max_input_channels']} input channels, "
            f"{device['max_output_channels']} output channels"
        )


def audio_callback(indata, frames, time, status, *, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    if status:
        print(f"Audio callback status: {status}")

    chunk = np.asarray(indata, dtype=np.int16).reshape(-1).copy()

    def _enqueue_chunk() -> None:
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    loop.call_soon_threadsafe(_enqueue_chunk)


def is_speech(chunk: np.ndarray, vad: webrtcvad.Vad, sample_rate: int) -> bool:
    if chunk.dtype != np.int16:
        raise TypeError("chunk must have dtype int16")
    if chunk.ndim != 1:
        raise ValueError("chunk must be a 1D int16 array")

    frame_ms = int(round(len(chunk) * 1000 / sample_rate))
    if len(chunk) * 1000 % sample_rate != 0 or frame_ms not in SUPPORTED_VAD_FRAME_MS:
        raise ValueError("chunk length must be exactly 10, 20, or 30 ms for webrtcvad")

    return vad.is_speech(chunk.tobytes(), sample_rate)


async def collect_utterance(audio_queue: asyncio.Queue, vad: webrtcvad.Vad, config: dict) -> np.ndarray:
    sample_rate = int(config.get("sample_rate", 16000))
    blocksize = int(config.get("blocksize", 480))
    silence_threshold_chunks = int(config.get("silence_threshold_chunks", 12))
    padding_chunks = max(1, int(round((300 / 1000) * sample_rate / blocksize)))

    pre_roll = deque(maxlen=padding_chunks)
    utterance_chunks: list[np.ndarray] = []
    in_speech = False
    silent_chunks = 0

    while True:
        chunk = await audio_queue.get()
        if chunk.dtype != np.int16 or chunk.ndim != 1:
            raise ValueError("audio_queue must contain flattened int16 chunks")

        chunk_is_speech = is_speech(chunk, vad, sample_rate)

        if not in_speech:
            pre_roll.append(chunk)
            if chunk_is_speech:
                in_speech = True
                utterance_chunks.extend(pre_roll)
                silent_chunks = 0
            continue

        utterance_chunks.append(chunk)
        if chunk_is_speech:
            silent_chunks = 0
        else:
            silent_chunks += 1
            if silent_chunks >= silence_threshold_chunks:
                return np.concatenate(utterance_chunks).astype(np.int16, copy=False)


def transcribe(audio: np.ndarray, model: WhisperModel, *, language: str | None = None, beam_size: int | None = None) -> str:
    if audio.dtype != np.int16:
        raise TypeError("audio must have dtype int16")
    if audio.ndim != 1:
        raise ValueError("audio must be a 1D int16 array")

    float_audio = audio.astype(np.float32) / 32768.0
    transcribe_kwargs: dict[str, object] = {}
    if language is not None:
        transcribe_kwargs["language"] = language
    if beam_size is not None:
        transcribe_kwargs["beam_size"] = beam_size

    segments, _info = model.transcribe(float_audio, **transcribe_kwargs)
    return " ".join(segment.text for segment in segments).strip()


async def start_listening(config: dict, shutdown_event: asyncio.Event) -> AsyncGenerator[str, None]:
    sample_rate = int(config.get("sample_rate", 16000))
    blocksize = int(config.get("blocksize", 480))
    channels = int(config.get("channels", 1))
    mic_device_index = config.get("mic_device_index")
    vad_aggressiveness = int(config.get("vad_aggressiveness", 2))
    max_queue_size = int(config.get("max_queue_size", 32))
    whisper_model_size = str(config.get("whisper_model_size", "small"))
    whisper_language = config.get("whisper_language")
    whisper_beam_size = config.get("whisper_beam_size")

    vad = webrtcvad.Vad(vad_aggressiveness)
    model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
    loop = asyncio.get_running_loop()
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    def _callback(indata, frames, time, status) -> None:
        audio_callback(indata, frames, time, status, queue=audio_queue, loop=loop)

    stream_kwargs = {
        "samplerate": sample_rate,
        "blocksize": blocksize,
        "channels": channels,
        "dtype": "int16",
        "callback": _callback,
    }
    if mic_device_index is not None:
        stream_kwargs["device"] = mic_device_index

    with sd.InputStream(**stream_kwargs):
        while not shutdown_event.is_set():
            utterance_task = asyncio.create_task(collect_utterance(audio_queue, vad, config))
            done, _pending = await asyncio.wait(
                {utterance_task, shutdown_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if shutdown_task in done:
                utterance_task.cancel()
                try:
                    await utterance_task
                except asyncio.CancelledError:
                    pass
                break

            audio = utterance_task.result()
            transcript = transcribe(
                audio,
                model,
                language=whisper_language,
                beam_size=whisper_beam_size,
            )
            if transcript:
                yield transcript

