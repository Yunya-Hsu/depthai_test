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

with dai.Device(pipeline) as device:
    enc_queue = device.getOutputQueue(name="enc", maxSize=fps, blocking=True)

    output_container = av.open('video.mp4', 'w')
    stream = output_container.add_stream('hevc', rate=fps)
    stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds

    while True:
        key = cv2.waitKey(1)

        start = time.time()
        try:
            while True:
                data = enc_queue.get().getData()  # np.array
                packet = av.Packet(data)  # Create new packet with byte array

                packet.pts = int((time.time() - start) * 1000 * 1000)
                output_container.mux_one(packet)  # Mux the Packet into container

        except KeyboardInterrupt:
            pass

        output_container.close()
