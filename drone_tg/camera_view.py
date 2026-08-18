from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

import cv2
import numpy as np

TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"

frame = None

def callback(msg):

    global frame

    img = np.frombuffer(
        msg.data,
        dtype=np.uint8
    )

    img = img.reshape(
        (msg.height,
         msg.width,
         3)
    )

    frame = img


node = Node()

node.subscribe(
    Image,
    TOPIC,
    callback
)

while True:

    if frame is not None:

        cv2.imshow(
            "PX4 Camera",
            frame
        )

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
