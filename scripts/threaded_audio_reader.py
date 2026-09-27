"""Threaded, non-blocking audio reader with Noise Gate (RMS threshold).

This example uses `sounddevice` to run a non-blocking audio stream and
places audio chunks into a bounded queue. A consumer thread computes RMS and
applies a noise gate (RMS < 0.02) to drop quiet frames.

Install: `pip install sounddevice numpy`
"""

import argparse
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd


class ThreadedAudioReader:
    def __init__(self, samplerate=16000, channels=1, blocksize=1024, queue_size=8, rms_threshold=0.02):
        self.sr = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.q = queue.Queue(maxsize=queue_size)
        self.rms_threshold = rms_threshold
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # non-fatal, just log
            print("Audio status:", status, file=sys.stderr)
        try:
            # copy to ensure memory safety
            arr = indata.copy()
            if self.q.full():
                try:
                    self.q.get_nowait()
                except Exception:
                    pass
            self.q.put_nowait(arr)
        except queue.Full:
            pass

    def start(self):
        self.stream = sd.InputStream(samplerate=self.sr, channels=self.channels, blocksize=self.blocksize, callback=self._callback)
        self.stream.start()

    def read_frame(self, timeout=0.1):
        try:
            arr = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        # compute RMS
        rms = np.sqrt(np.mean(np.square(arr)))
        if rms < self.rms_threshold:
            return None
        return arr, float(rms)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


def demo(duration=10):
    reader = ThreadedAudioReader(samplerate=16000, channels=1, blocksize=1024, rms_threshold=0.02)
    reader.start()
    t0 = time.time()
    try:
        while time.time() - t0 < duration:
            out = reader.read_frame(timeout=0.5)
            if out is None:
                # quiet or no data
                continue
            arr, rms = out
            print(f"Heard audio frame, RMS={rms:.4f}")
    finally:
        reader.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, default=10)
    args = p.parse_args()
    demo(args.seconds)
