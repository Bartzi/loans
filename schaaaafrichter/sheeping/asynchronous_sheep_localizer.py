import multiprocessing
import queue
import time

from sheeping.sheep_localizer import SheepLocalizer


class AsynchronousSheepLocalizer(SheepLocalizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stop_localization = multiprocessing.Event()
        self.localization_queue = multiprocessing.Queue(maxsize=1)
        self.image_queue = multiprocessing.Queue(maxsize=1)
        self.localization_process = None

    def start_localization_worker(self):
        def worker():
            if not self.initialized:
                self.build_model()
            while True:
                if self.stop_localization.is_set():
                    break
                try:
                    frame = self.localization_queue.get(timeout=5)
                except queue.Empty:
                    continue

                start_time = time.time()
                bboxes, scores = self.localize(frame)
                end_time = time.time()
                fps = 1.0 / (end_time - start_time)
                self.image_queue.put((bboxes, scores, fps))

        self.localization_process = multiprocessing.Process(target=worker)
        self.localization_process.start()

    @staticmethod
    def empty_queue(q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass  # we are done with emptying this queue

    def shutdown(self):
        self.stop_localization.set()
        self.empty_queue(self.localization_queue)
        self.empty_queue(self.image_queue)
        self.localization_process.join()

