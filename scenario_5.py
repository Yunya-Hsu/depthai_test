import av
import cv2
import depthai as dai
from fractions import Fraction
from FPS import FPS
import time
import numpy as np

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

stereo_depth = pipeline.create(dai.node.StereoDepth)
stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo_depth.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
stereo_depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
mono_left.out.link(stereo_depth.left)
mono_right.out.link(stereo_depth.right)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo_depth.disparity.link(xout_depth.input)

with dai.Device(pipeline) as device:
    fps_calculator = FPS()

    enc_queue = device.getOutputQueue(name="enc", maxSize=fps, blocking=True)
    depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    output_container = av.open('video.mp4', 'w')
    stream = output_container.add_stream('hevc', rate=fps)
    stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds
    start = time.time()

    try:
        while True:
            key = cv2.waitKey(1)

            in_depth = depth_queue.tryGet()
            if in_depth is not None:
                fps_calculator.update()
                frame = in_depth.getFrame()

                frame = (frame * (255 / stereo_depth.initialConfig.getMaxDisparity())).astype(np.uint8)
                fps_calculator.draw(frame, orig=(50, 50), size=1, color=(240, 180, 100)) # around 30 fps
                cv2.imshow("depth", frame)

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
