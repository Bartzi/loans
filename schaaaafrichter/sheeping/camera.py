import cv2


class Camera:

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.capture_device = None
        self.last_image = None

    def __enter__(self):
        self.capture_device = cv2.VideoCapture(self.camera_id)
        return self

    def get_frame(self):
        ret, frame = self.capture_device.read()
        if not ret:
            if self.last_image is None:
                raise RuntimeError("Camera not ready?")
            return self.last_image
        self.last_image = frame
        return frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capture_device.release()
