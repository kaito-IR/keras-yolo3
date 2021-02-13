import sys
import argparse
import pyrealsense2 as rs
import  subprocess as sp
import numpy as np
from yolo import YOLO, detect_video
from PIL import Image
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    FLAGS = parser.parse_args()
    detect_video(YOLO(**vars(FLAGS)))