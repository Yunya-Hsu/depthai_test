import cv2
import depthai as dai
from FPS import FPS

fps = 60

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setFps(fps)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input) # set preview for xout

with dai.Device(pipeline) as device:
    fps_calculator = FPS()
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    while True:
        key = cv2.waitKey(1)

        in_rgb = rgb_queue.get()
        fps_calculator.update()

        frame = in_rgb.getCvFrame()
        fps_calculator.draw(frame, orig=(50, 50), size=1, color=(240, 180, 100)) # FPS around 60
        cv2.imshow("rgb", frame)

        if key == ord('q'):
            break
