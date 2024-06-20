from Detector import *

import os

def main():

    videoPath = "test_videos/street1.mp4"

    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("madel_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    Detector(videoPath, configPath, modelPath, classesPath)
    Detector.onVideo()

    if __name__ =='__main__':
        main()
