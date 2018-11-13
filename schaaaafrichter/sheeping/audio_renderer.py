import threading

from time import sleep

import simpleaudio


class Baaaer:

    def __init__(self, audio_file='sheeping/sheep.wav'):
        self.shutdown_signal = threading.Event()
        self.play_event = threading.Event()
        self.song = simpleaudio.WaveObject.from_wave_file(audio_file)
        self.baa_thread = None
        self.enabled = False
        self.init()

    def init(self):
        self.baa_thread = threading.Thread(target=self.play_worker)
        self.baa_thread.start()

    def shutdown(self):
        self.shutdown_signal.set()
        if self.baa_thread is not None:
            self.baa_thread.join()

    def play_worker(self):
        while True:
            if self.shutdown_signal.is_set():
                break
            if self.play_event.wait(1):
                playback = self.song.play()
                playback.wait_done()
                self.play_event.clear()
            sleep(1)

    def baaa(self):
        if self.enabled:
            self.play_event.set()
