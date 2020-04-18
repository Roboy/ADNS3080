#!/usr/bin/env python
#
# Copyright DIY Drone: https://github.com/diydrones/ardupilot/tree/5ddbcc296dd6dd9ac9ed6316ac3134c736ae8a78/libraries/AP_OpticalFlow/examples/ADNS3080ImageGrabber
# File: ADNS3080ImageGrabber.py
# Modified by Kristian Sloth Lauszus

from serial import Serial, SerialException
from threading import Timer

import serial.tools.list_ports
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np

def serial_ports():
    devices = []
    for port in list(serial.tools.list_ports.comports()):
        if port.hwid != 'n/a':  # Check if it is a physical port
            devices.append(port.device)
    if devices:
        print('Serial ports:')
        for device in devices:
            print('  ' + device)
    else:
        print('No serial ports found')
    return devices


class OpticalFlow:
    comPortStr = "/dev/ttyACM0"
    baudrateStr = "115200"
    grid_size = 15
    num_pixels = 30
    ser = None
    t = None
    pixel_dictionary = {}
    feature_params = []
    lk_params = []
    color = []
    prevFrame,frame = [],[]
    p0 = None
    mask = []
    img_scaled = None
    img_scaled_flow = None
    iterations = 0
    max_iterations = 1000

    def __init__(self):
        ports = serial_ports()
        if not ports:
            ports = ['No serial ports found']

        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 10,
                               qualityLevel = 0.3,
                               minDistance = 3,
                               blockSize = 3 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (30,30),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        self.color = np.random.randint(0,255,(100,3))

        # Take first frame and find corners in it
        self.frame = np.zeros((self.num_pixels,self.num_pixels,1), np.float32)
        self.prevFrame = np.zeros((self.num_pixels,self.num_pixels,1), np.float32)

        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(self.prevFrame)

        self.img = self.frame.astype(np.uint8)
        self.prevImg = self.prevFrame.astype(np.uint8)

    def __del__(self):
        self.close()

    def close(self):
        self.close_serial()

    def open(self):
        # Close the serial port
        self.close_serial()

        # Open the serial port
        try:
            self.ser = Serial(port=self.comPortStr, baudrate=self.baudrateStr, timeout=.1)
            print("Serial port '" + self.comPortStr + "' opened at " + self.baudrateStr)
            if self.ser.isOpen():
                try:
                    self.read_from_serial()
                    self.img = self.frame.astype(np.uint8).copy()
                    self.img_scaled = cv2.resize(self.img,None,fx=10, fy=10, interpolation = cv2.INTER_CUBIC)
                    flow.img_scaled_flow = flow.img_scaled.copy()
                except (IOError, TypeError):
                    pass
        except SerialException:
            print("Failed to open serial port '" + self.comPortStr + "'")

    def close_serial(self):
        if self.ser and self.ser.isOpen():
            try:
                self.ser.close()
                print("Closed serial port")
            except SerialException:
                pass  # Do nothing

    def calc(self):
        if self.t:
            self.stop_read_loop()

        if self.ser.isOpen():
            try:
                self.read_from_serial()
            except (IOError, TypeError):
                pass

            if self.iterations > self.max_iterations:
                self.p0 = cv2.goodFeaturesToTrack(self.frame, mask = None, **self.feature_params)
                self.img_scaled_flow = self.img_scaled.copy()
                self.iterations = 0


            if self.p0 is not None:
                # for p in self.p0:
                #     self.display_pixel(int(p[0][1]),int(p[0][0]),255)
                # calculate optical flow
                self.img = self.frame.astype(np.uint8).copy()
                self.prevImg = self.prevFrame.astype(np.uint8)

                self.img_scaled = cv2.resize(self.img,None,fx=10, fy=10, interpolation = cv2.INTER_CUBIC)
                cv2.imshow('img',self.img_scaled)
                cv2.waitKey(1)

                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prevImg, self.img, self.p0, None, **self.lk_params)

                if p1 is not None:
                    # for p in p1:
                    #     self.display_pixel(int(p[0][1]),int(p[0][0]),255)

                    # Select good points
                    good_new = p1[st==1]
                    good_old = self.p0[st==1]
                    # print(good_new)
                    # print(good_old)

                    # draw the tracks
                    for i,(new,old) in enumerate(zip(good_new,good_old)):
                        a,b = new.ravel()*10
                        d,c = old.ravel()*10
                        self.img_scaled_flow = cv2.circle(self.img_scaled_flow,(a,b),5,self.color[i].tolist(),-1)

                    cv2.imshow('flow',self.img_scaled_flow)
                    cv2.waitKey(1)
                #
                    # # Now update the previous points
                    self.p0 = good_new.reshape(-1,1,2)
            else:
                print("reset")
                self.p0 = cv2.goodFeaturesToTrack(self.frame, mask = None, **self.feature_params)
            # print(self.p0)
            self.prevFrame = self.frame.copy()
            self.iterations = self.iterations+1

    def read_from_serial(self):
        while self.ser and self.ser.isOpen() and self.ser.inWaiting() > 0:
            # Process the line read
            line = self.ser.readline()
            if line.find("start") == 0:
                # print('Started reading image')
                pixels = self.ser.read(self.num_pixels * self.num_pixels)
                if len(pixels) == self.num_pixels * self.num_pixels:
                    for row in range(self.num_pixels):
                        # print(row)
                        col = 0
                        for p in pixels[row * self.num_pixels:(row + 1) * self.num_pixels]:
                            try:
                                colour = ord(p)
                            except TypeError:
                                colour = 0
                            # print('Colour', colour)
                            self.frame[self.num_pixels - 1 - row, self.num_pixels - 1 - col] = colour
                            col += 1
                    # print('Done reading image')
                else:
                    print(len(pixels))
                    # print("Bad line: " + pixels)
            else:
                # Display the line if we couldn't understand it
                print('Error while processing string:', line)
                self.ser.flushInput()  # Flush the input, as this was likely caused by a timeout

    # def display_default_image(self):
    #     # Display the grid
    #     for x in range(self.num_pixels):
    #         for y in range(self.num_pixels):
    #             colour = x * y / 3.53
    #             self.display_pixel(x, y, colour)
    #
    # def display_pixel(self, x, y, colour):
    #     if 0 <= x < self.num_pixels and 0 <= y < self.num_pixels:
    #         # Find the old pixel if it exists and delete it
    #         if x+y*self.num_pixels in self.pixel_dictionary:
    #             old_pixel = self.pixel_dictionary[x+y*self.num_pixels]
    #             self.canvas.delete(old_pixel)
    #             del old_pixel
    #
    #         fill_colour = "#%02x%02x%02x" % (int(colour), int(colour), int(colour))
    #         # Draw a new pixel and add to pixel_array
    #         new_pixel = self.canvas.create_rectangle(x*self.grid_size, y*self.grid_size, (x+1)*self.grid_size,
    #                                                  (y+1)*self.grid_size, fill=fill_colour)
    #         self.pixel_dictionary[x+y*self.num_pixels] = new_pixel

flow = OpticalFlow()
flow.open()
while True:
    flow.calc()
