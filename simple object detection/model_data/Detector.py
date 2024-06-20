import cv2

import numpy as np
import time

class Detector:
    def __inti__(self, videoPath, configPath, modelPath, classesPath):

        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()


    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0,"__Background__")

        self.colorList =np.random.uniform(low=0, high=255, size=(len(self.classesList),3))

        print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened()==False):
            print("Error opening file...")
            return
        
        (success, image ) = cap.read()

        while success:
            classLabelIDs, confidences, bboxs =self.net.detect(image, confThreshold = 0.5)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float,confidences))

            bboxsIdx =cv2.dnn.NMSBoxes(bboxs,confidences, score_threshold= 0.5, nms_threshold= 0.2)

            if len(bboxsIdx) != 0:
                for i in range(0,len(bboxsIdx)):


                    bbox = bboxs[np.squeeze(bboxsIdx[i])] 
                    classConfidence = confidences[np.squeeze(bboxsIdx[i])] 
                    classLabelIDs = np.squeeze(classLabelIDs[np.squeeze(bboxsIdx[i])]) 
                    classLabel = self.classesList[classLabelIDs]
                    classColor = [int(c) for c in self.colorList[classLabelIDs]]

                    displayText = "{}:{:.4f}".formate(classLabel, classConfidence)




                    x,y,w,h = bbox

                    cv2.rectangle(image,(x,y),(x+w , y+h), color=classColor, thickness=1)
                    cv2.putText(image, displayText,(x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, classColor, 2)

            cv2.imshow("Result", image)

            key = cv2.waitkey(1) & 0xFF
            if key == ord("q"):
                break 

            (success, image) = cap.read()

        cv2.destroyAllWindows()






