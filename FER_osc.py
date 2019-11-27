from keras.models import model_from_json
import numpy as np
import cv2
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import math

from pythonosc import udp_client

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        # print("Model loaded from disk")
        # self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return self.preds#FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("fps")
args = parser.parse_args()
cap = cv2.VideoCapture(os.path.abspath(args.source) if not args.source == 'webcam' else 0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap.set(cv2.CAP_PROP_FPS, int(args.fps))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

DNN = "TF"
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

def getdata():
    _, fr = cap.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    frameOpencvDnn = fr.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
 
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            # print("has face!",x1,y1,x2,y2)
            bboxes.append([x1, y1, x2, y2])
    return bboxes, fr, gray

def start_app(cnn):
    while cap.isOpened():
        faces, fr, gray_fr = getdata()
        for (x, y, x2, y2) in faces:
            fc = gray_fr[y:y2, x:x2]
            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            emotion = FacialExpressionModel.EMOTIONS_LIST[np.argmax(pred)]
            for idx,i in enumerate(FacialExpressionModel.EMOTIONS_LIST):
                color = (211, 211, 211) if pred[0][idx] < 0.01 else (0, 255, 0)
                emotion_score = "{}: {}".format(i, "{:.2f}".format(pred[0][idx]) if pred[0][idx] > 0.01 else "")
                cv2.putText(fr, emotion_score, (x2 + 5, y + 15 + idx*18), font, 0.5, color, 1, cv2.LINE_AA)
            cv2.rectangle(fr, (x, y), (x2, y2), (255, 0, 0), 2)

            client.send_message("/found",1)
            client.send_message("/face",[x,y,x2-x,y2-y])
            client.send_message("/emotion", emotion)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Facial Emotion Recognition', fr)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ip = "127.0.0.1"
    port = 12345
    client = udp_client.SimpleUDPClient(ip, port)

    model = FacialExpressionModel("model.json", "weights.h5")
    start_app(model)
