import cv2
import numpy as np
import argparse

argParser = argparse.ArgumentParser()

# Show only bounding boxes
argParser.add_argument("-b", "--blank", action='store_true')
argParser.add_argument("-y", "--height", type=int, default=720, help="Camera viewport height in pixels")
argParser.add_argument("-x", "--width", type=int, default=1280, help="Camera viewport width in pixels")
argParser.add_argument("-fs", "--frameskips", type=int, default=4, help="Ammount of frames to skip, larger number reduces cpu usage")
# argParser.add_argument("-f", "--fps", type=int, default=30, help="fps limit")
argParser.add_argument("-f", "--fullscreen", type=bool, default=False, help="full screen")
args = vars(argParser.parse_args())


windowName = "Video"
# Skipping frames reduces cpu-usage dramatically
resizeHeight = 480
frameSkips = args["frameskips"]
cameraHeight = args["height"]
cameraWidth = args["width"]
blank = args["blank"]

try:
    cascadePath = "venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
except BaseException:
    print("Invalid or missing cascade data path.")
    exit()


def Render(frame, faces):

    color = (int(0), int(255), int(0))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, [x, y], [x + w, y + h], tuple(color), 2)

    if (args["fullscreen"]):
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow(windowName, frame)


def faceDetector(frame):
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces


def main():

    videoCapture = cv2.VideoCapture(0)
   # videoCapture.set(3, resizeHeight)
   # videoCapture.set(4, resizeWidth)

    if (videoCapture.isOpened() is False):
        print("Error connecting to camera.")
        exit()
    else:
        print("Camera found.")

    if (blank):
        print("Blank mode enabled.")

    returnCode, frame = videoCapture.read()

    # Check if first frame is valid:
    if returnCode == True:
        height, width, channels = frame.shape
        print(f'Resolution:\t{height} x {width} ')
        # frameResizeScale = float(height)/resizeHeight
    else:
        print("Unable to read frame")
        exit()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Track FPS
    time = cv2.getTickCount()
    count = 0

    while True:
        if count == 0:
            time = cv2.getTickCount()

        returnCode, frame = videoCapture.read()

        # resize frame to process faster - pointless
        # frameSmall = cv2.resize(frame, None, fx=1.0/frameResizeScale,
        # fy=1.0/frameResizeScale, interpolation=cv2.INTER_LINEAR)
        # frameSmallGray = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2GRAY)

        if count % frameSkips == 0:
            faces = faceDetector(frame)

        # 27 = esc
        if (cv2.waitKey(1) & 0xFF == 27):
            break

        count += 1
        if (count == 100):
            time = (cv2.getTickCount() - time) / cv2.getTickFrequency()
            fps = 100.0 / time

            print(f'FPS: {fps}', end='\r')

            count = 0

        # Blank frame for showing only bounding boxes
        if (blank):
            blankFrame = np.zeros((height, width, 3), np.uint8)
            Render(blankFrame, faces)
        else:
            Render(frame, faces)

    print("\nGoodbye.")
    videoCapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
