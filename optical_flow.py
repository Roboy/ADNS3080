#!/usr/bin/env python
#
# Copyright DIY Drone: https://github.com/diydrones/ardupilot/tree/5ddbcc296dd6dd9ac9ed6316ac3134c736ae8a78/libraries/AP_OpticalFlow/examples/ADNS3080ImageGrabber
# File: ADNS3080ImageGrabber.py
# Modified by Kristian Sloth Lauszus

from serial import Serial, SerialException
from Tkinter import Tk, Frame, StringVar, Button, Canvas, OptionMenu
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


class App:
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

    def __init__(self, master):
        # Set main window's title
        master.title("ADNS3080ImageGrabber")

        frame = Frame(master)
        frame.grid(row=0, column=0)

        self.comPortStr = StringVar()
        ports = serial_ports()
        if not ports:
            ports = ['No serial ports found']
        self.comPortStr.set(ports[0])  # Set first port as default

        comports = apply(OptionMenu, (frame, self.comPortStr) + tuple(ports))
        comports.grid(row=0, column=0)

        self.baudrateStr = StringVar()
        self.baudrateStr.set('115200')

        baudrates = apply(OptionMenu, (frame, self.baudrateStr) + tuple(Serial.BAUDRATES))
        baudrates.grid(row=0, column=1)

        button = Button(frame, text="Open", command=self.open)
        button.grid(row=0, column=2)

        button = Button(frame, text="Close", command=self.close)
        button.grid(row=0, column=3)

        self.canvas = Canvas(master, width=self.grid_size*self.num_pixels, height=self.grid_size*self.num_pixels)
        self.canvas.grid(row=1)

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

    def __del__(self):
        self.close()

    def close(self):
        self.stop_read_loop()
        self.close_serial()

    def open(self):
        # Close the serial port
        self.close_serial()

        # Open the serial port
        try:
            self.ser = Serial(port=self.comPortStr.get(), baudrate=self.baudrateStr.get(), timeout=.1)
            print("Serial port '" + self.comPortStr.get() + "' opened at " + self.baudrateStr.get())
            self.read_loop()  # Read from serial port
        except SerialException:
            print("Failed to open serial port '" + self.comPortStr.get() + "'")

    def close_serial(self):
        if self.ser and self.ser.isOpen():
            try:
                self.ser.close()
                print("Closed serial port")
            except SerialException:
                pass  # Do nothing

    def read_loop(self):
        if self.t:
            self.stop_read_loop()

        if self.ser.isOpen():
            try:
                self.read_from_serial()
            except (IOError, TypeError):
                pass

            if self.p0 is not None:
                # for p in self.p0:
                #     self.display_pixel(int(p[0][1]),int(p[0][0]),255)
                # calculate optical flow
                img = self.frame.astype(np.uint8)
                prevImg = self.prevFrame.astype(np.uint8)


                img_scaled = cv2.resize(img,None,fx=10, fy=10, interpolation = cv2.INTER_CUBIC)
                cv2.imshow('frame',img_scaled)
                # cv2.waitKey(1)

                p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg, img, self.p0, None, **self.lk_params)

                if p1 is not None:
                    for p in p1:
                        self.display_pixel(int(p[0][1]),int(p[0][0]),255)

                    # Select good points
                    good_new = p1[st==1]
                    good_old = self.p0[st==1]
                    print(good_new)
                    print(good_old)
                #
                #     # draw the tracks
                #     for i,(new,old) in enumerate(zip(good_new,good_old)):
                #         a,b = new.ravel()
                #         c,d = old.ravel()
                #         self.mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                #         self.frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                #     img = cv2.add(self.frame,self.mask)
                #
                #     cv2.imshow('frame',img)
                #     k = cv2.waitKey(30) & 0xff
                #     if k == 27:
                #         return
                #
                    # # Now update the previous points
                    self.p0 = good_new.reshape(-1,1,2)
                else:
                    self.p0 = cv2.goodFeaturesToTrack(self.frame, mask = None, **self.feature_params)
            else:
                self.p0 = cv2.goodFeaturesToTrack(self.frame, mask = None, **self.feature_params)

            self.prevFrame = self.frame

        self.t = Timer(0.0, self.read_loop)
        self.t.start()

    def stop_read_loop(self):
        try:
            self.t.cancel()
        except AttributeError:
            print("Failed to cancel timer")

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
                            self.display_pixel(self.num_pixels - 1 - row, self.num_pixels - 1 - col, colour)
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

    def display_default_image(self):
        # Display the grid
        for x in range(self.num_pixels):
            for y in range(self.num_pixels):
                colour = x * y / 3.53
                self.display_pixel(x, y, colour)

    def display_pixel(self, x, y, colour):
        if 0 <= x < self.num_pixels and 0 <= y < self.num_pixels:
            # Find the old pixel if it exists and delete it
            if x+y*self.num_pixels in self.pixel_dictionary:
                old_pixel = self.pixel_dictionary[x+y*self.num_pixels]
                self.canvas.delete(old_pixel)
                del old_pixel

            fill_colour = "#%02x%02x%02x" % (int(colour), int(colour), int(colour))
            # Draw a new pixel and add to pixel_array
            new_pixel = self.canvas.create_rectangle(x*self.grid_size, y*self.grid_size, (x+1)*self.grid_size,
                                                     (y+1)*self.grid_size, fill=fill_colour)
            self.pixel_dictionary[x+y*self.num_pixels] = new_pixel

root = Tk()

# Create main display
app = App(root)
app.display_default_image()

print("Entering main loop")
root.mainloop()
app.close()
print("Exiting")
