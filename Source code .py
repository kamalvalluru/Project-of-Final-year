
import numpy as np
import argparse
import serial
import time

prevname="NIL"

onestring="1"
zerostring="0"
sentcommand=0

objectthere=0
resetbit=0

runIs = True
print('Connecting to Arduino........')
try:
    arduino = serial.Serial(port='COM4', baudrate=9600)
except:
    print(' Failed to Connected with Arduino! \n--------------------------------- \n Connect Arduino with correct port\n--------------------------------- \n Windows: COM port\n--------------------------------- \n Linux or Mac dev/tty or " Google it for Mac Or Linux"')
    print("---------------------------------\n Exiting ....")
    runIs = False
else:
    # print(runIs)
    print("successfully connected to The Arduino, ")

command="0"
sendsecondscount=0    


try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python" subdirectory if required)')

inWidth = 400
inHeight = 400
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for object Detection '
                    ' with deep learning TensorFlow frameworks.')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--prototxt", default="ssd_mobilenet_v1_coco.pbtxt")
    parser.add_argument("--weights", default="frozen_inference_graph.pb")
    parser.add_argument("--num_classes", default=90, type=int)
    parser.add_argument("--thr", default=0.4, type=float, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()

    if args.num_classes == 90:
        net = cv.dnn.readNetFromTensorflow(args.weights, args.prototxt)
        swapRB = True
        classNames = { 0: 'background',44: 'bottle', 62: 'chair', 77: 'cell phone', 84: 'book', 73: 'laptop', 74: 'mouse', 76: 'keyboard' }

    if args.video:
        cap = cv.VideoCapture(args.video)
    else:
        cap = cv.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
        net.setInput(blob)
        detections = net.forward()

        cols = frame.shape[1]
        rows = frame.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        frame = frame[y1:y2, x1:x2]

        cols = frame.shape[1]
        rows = frame.shape[0]

        for i in range(detections.shape[2]):
            objectthere=1
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)

                
                if class_id in classNames:
                    #print(class_id)

                    cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    if class_id==44 :
                        print("==== BOTTEL DETECTED ============")
                        print("==========SENDING COMMAND:",onestring)
                        arduino.write(onestring.encode())
                        time.sleep(7)
                        arduino.flush()
                    else:
                        print("===== IRRELEVANT OBJECT =====")
                        print("==========SENDING COMMAND:",zerostring)
                        arduino.write(zerostring.encode())
                        time.sleep(10)
                        arduino.flush()

        cv.imshow("detections", frame)
        if cv.waitKey(1) >= 0:
            break
