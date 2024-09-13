import av
import cv2
import depthai as dai
from fractions import Fraction
from FPS import FPS
import time

fps = 60

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setFps(fps)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

video_enc = pipeline.create(dai.node.VideoEncoder)
video_enc.setDefaultProfilePreset(fps, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam_rgb.video.link(video_enc.input)

xout_enc = pipeline.create(dai.node.XLinkOut)
xout_enc.setStreamName("enc")
video_enc.bitstream.link(xout_enc.input)

mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setCamera("left")
mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setCamera("right")

xout_left = pipeline.create(dai.node.XLinkOut)
xout_left.setStreamName("left")
mono_left.out.link(xout_left.input)
xout_right = pipeline.create(dai.node.XLinkOut)
xout_right.setStreamName("right")
mono_right.out.link(xout_right.input)


with dai.Device(pipeline) as device:
    fps_calculator = FPS()

    enc_queue = device.getOutputQueue(name="enc", maxSize=fps, blocking=True)
    left_queue = device.getOutputQueue(name="left", maxSize=1, blocking=False)
    right_queue = device.getOutputQueue(name="right", maxSize=1, blocking=False)

    output_container = av.open('video.mp4', 'w')
    stream = output_container.add_stream('hevc', rate=fps)
    stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds
    start = time.time()

    try:
        while True:
            key = cv2.waitKey(1)

            in_left = left_queue.tryGet()
            if in_left is not None:
                fps_calculator.update()
                left_frame = in_left.getCvFrame()
                fps_calculator.draw(left_frame, orig=(50, 50), size=1, color=(240, 180, 100)) # around 30 fps
                cv2.imshow("mono_left", left_frame)

            in_right = right_queue.tryGet()
            if in_right is not None:
                cv2.imshow("mono_right", in_right.getCvFrame())

            while enc_queue.has():
                data = enc_queue.get().getData()
                packet = av.Packet(data)
                packet.pts = int((time.time() - start) * 1000 * 1000)
                output_container.mux_one(packet)

            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        output_container.close()
        cv2.destroyAllWindows()
