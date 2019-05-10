# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Apr 10 2019)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
from PIL import Image
import numpy as np
import cv2
import os
import math
import sys
from scipy import ndimage

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

###########################################################################
## Class MyFrame3
###########################################################################
class BandRejectFilter:
    image = None
    filter = None
    cutoff1 = None
    cutoff2 = None
    order = None
    output = None

    def __init__(self, image, filter_name, cutoff1 =170, cutoff2=250, order = 0):
        self.image = image
        if filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter
        elif filter_name == 'gaussian_BRF':
            self.filter = self.get_gaussian_BRF

        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.order = order


    def get_gaussian_BRF(self, shape, cutoff1, cutoff2):
        height = shape[0]
        width = shape[1]
        gaussianBRF = np.zeros((height, width))
        gaussianImageL = self.get_gaussian_low_pass_filter(shape, cutoff1)
        gaussianImageH = self.get_gaussian_high_pass_filter(shape, cutoff2)
        for u in range(height):
            for v in range(width):
                gaussianBRF[u, v] = 1 - (gaussianImageL[u, v] * gaussianImageH[u, v])
        return gaussianBRF

    def get_gaussian_low_pass_filter(self, shape, cutoff1):
        height = shape[0]
        width = shape[1]
        m = height / 2
        n = width / 2
        gaussianImageL = np.zeros((height, width))
        for u in range(height):
            for v in range(width):
                dist = self.getDist(u, v, m, n)
                gaussianImageL[u,v] = math.pow(math.e, -math.pow(dist,2)/(2*math.pow(cutoff1,2)))
        return gaussianImageL

    def get_gaussian_high_pass_filter(self, shape, cutoff2):
        #Hint: May be one can use the low pass filter function to get a high pass mask
        height = shape[0]
        width = shape[1]
        gaussianImageL = self.get_gaussian_low_pass_filter(shape, cutoff2)
        gaussianImageH = np.zeros((height, width))
        for u in range(height):
            for v in range(width):
                gaussianImageH[u,v] = 1 - gaussianImageL[u,v]
        return gaussianImageH

    def post_process_image(self, image):
        imageMin = np.min(image)
        imageMax = np.max(image)
        newImage = np.uint8(255 / (imageMax - imageMin) * (image - imageMin) + .5)
        return newImage


    def filtering(self):


        # 1 and 2
        fftImage = np.fft.fft2(self.image)
        shiftImage = np.fft.fftshift(fftImage)
        # 3
        mask = self.filter(self.image.shape, self.cutoff1, self.cutoff2)
        # 4, 5 and 6
        filteredImage = shiftImage * mask
        inverseShiftImage = np.fft.ifftshift(filteredImage)
        inverseShiftImage = np.fft.ifft2(inverseShiftImage)
        # 7
        magnitudeDFT = np.log(np.abs(shiftImage))
        mask = np.log(np.abs(mask))
        mask = np.abs(mask)
        mask = 1000 * mask
        print(mask)

        magnitudeDFT = magnitudeDFT*20
        #magnitudeDFT = np.log(abs(shiftImage))
        #magnitudeDFT = np.log(cv2.magnitude(shiftImage[:,:,0], shiftImage[:,:,1]))
        #magnitudeDFT, angle = cv2.cartToPolar(np.real(filteredImage), np.imag(filteredImage))
        magnitudeImage = np.abs(inverseShiftImage)
        magnitudeFilteredfft = np.log(np.abs(inverseShiftImage))
        # 8
        #magnitudeDFT = self.post_process_image(magnitudeDFT)
        magnitudeFilteredfft = self.post_process_image(magnitudeFilteredfft)
        finalFilteredImage = self.post_process_image(magnitudeImage)

        return [finalFilteredImage, magnitudeDFT, mask]

        #mask = mask.astype(np.uint8)
        #shiftImage = shiftImage.astype(np.complex_)
        #fftImage = fftImage.astype(np.uint8)
        #return [finalFilteredImage, shiftImage, magnitudeImage]


    def getDist(self, u, v, m, n):
        a = math.pow((u-m), 2)
        b = math.pow((v-n), 2)
        return math.sqrt(a+b)

class adaptive_filter:



    def __init__(self, image, adaptive_filter_name):
        self.image=image
        #self.filter_h = 3
        #self.filter_w = 3
        if adaptive_filter_name=='reduction':
            self.filter=self.get_adaptive_reduction_filter
        elif adaptive_filter_name=='median':
            self.filter=self.get_adaptive_median_filter
    local_count = 0
    local_zero = 0
    local_reduced_intensity = 0

    def reduction_filter(self,image_slice, value, filter_h, filter_w, variance):
        v = variance
        if v == 0:
            return value
        else:
             local_var = int(ndimage.variance(image_slice))
             m = image_slice.mean()
             if variance == local_var:
                 self.local_count = self.local_count + 1
                 return np.uint8(m)
             elif int(local_var) == 0:
                 self.local_zero = self.local_zero + 1
                 return value
             else:
                 self.local_reduced_intensity = self.local_reduced_intensity + 1
                 return (value - (v/local_var)*(value - m))


    def median_filter(self, image_slice, value, filter_h, filter_w,max_int):
            w = image_slice.shape[1]
            for i in range(w):
                max_int = max_int+1
            vert_start = int(1/2*(max_int-filter_h))
            horiz_start = int(1/2*(max_int-filter_w))
            vert_end = int(1/2*(max_int+filter_h))
            horiz_end = int(1/2*(max_int+filter_w))
            windowS = image_slice[vert_start:vert_end , horiz_start:horiz_end]
            try:
                zmed= np.median(windowS)
            except ValueError:
                zmed = 0
            try:
                zmin = np.min(windowS)
            except ValueError:
                zmin =  0
            try:
                zmax = np.max(windowS)
            except ValueError:
                zmax = 0
            if zmed>zmin and zmed<zmax:
                if value>zmin and value<zmax:
                    return value
                else:
                    return zmed
            else:
                if (filter_h + 2)> max_int or (filter_w+ 2)> max_int:
                    return zmed
                else:
                    return self.median_filter(image_slice, value, filter_h+2, filter_w+2, max_int)


    def get_adaptive_reduction_filter(self, image):
        filter_h = 3
        img_var = int(ndimage.variance(image))
        filter_w = 3
        pad_h = int((filter_h - 1) / 2)
        pad_w = int((filter_w - 1) / 2)
        pad_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
        height, width = image.shape
        filtered_image = np.zeros((height, width))
        for u in range(height):
            for v in range(width):
                temp_image=pad_img[u:(u+3),v:(v+3)]
                filtered_image[u][v] = self.reduction_filter(temp_image,pad_img[u][v],filter_h,filter_w,img_var)
        return filtered_image

    def get_adaptive_median_filter(self, image):
        filter_h = 3
        img_var = ndimage.variance(image)
        filter_w = 3
        max_int = 0
        pad_h = int((filter_h - 1) / 2)
        pad_w = int((filter_w - 1) / 2)
        pad_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
        height, width = image.shape
        filtered_image = np.zeros((height, width))
        for u in range(height):
            for v in range(width):
                temp = pad_img[u:(u+3),v:(v+3)]
                filtered_image[u][v] = self.median_filter(image,pad_img[u][v],filter_h,filter_w,max_int)
        return filtered_image

    def filtering(self):
        if self.filter==self.get_adaptive_reduction_filter:
            filtered_image=self.get_adaptive_reduction_filter(self.image)
        elif self.filter==self.get_adaptive_median_filter:
            filtered_image=self.get_adaptive_median_filter(self.image)
        return filtered_image

class RectangleSelectImagePanel(wx.Panel):
    # majority of code in this class was found on the internet
    def __init__(self, parent, pathToImage=None):
        # Initialise the parent
        wx.Panel.__init__(self, parent)

        # Intitialise the matplotlib figure
        self.figure = plt.figure()

        # Create an axes, turn off the labels and add them to the figure
        self.axes = plt.Axes(self.figure,[0,0,1,1])
        self.axes.set_axis_off()
        self.figure.add_axes(self.axes)

        # Add the figure to the wxFigureCanvas
        self.canvas = FigureCanvas(self, -1, self.figure)

        self.Image = None

        # Initialise the rectangle
        self.rect = Rectangle((0,0), 1, 1, facecolor='None', edgecolor='green')
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.axes.add_patch(self.rect)

        # Sizer to contain the canvas
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 3, wx.ALL)
        self.SetSizer(self.sizer)
        self.Fit()

        # Connect the mouse events to their relevant callbacks
        self.canvas.mpl_connect('button_press_event', self._onPress)
        self.canvas.mpl_connect('button_release_event', self._onRelease)
        self.canvas.mpl_connect('motion_notify_event', self._onMotion)

        # Lock to stop the motion event from behaving badly when the mouse isn't pressed
        self.pressed = False

        # If there is an initial image, display it on the figure
        if pathToImage is not None:
            self.setImage(pathToImage)


    def _onPress(self, event):
        ''' Callback to handle the mouse being clicked and held over the canvas'''

        # Check the mouse press was actually on the canvas
        if event.xdata is not None and event.ydata is not None:

            # Upon initial press of the mouse record the origin and record the mouse as pressed
            self.pressed = True
            self.rect.set_linestyle('dashed')
            self.x0 = event.xdata
            self.y0 = event.ydata


    def _onRelease(self, event):
        '''Callback to handle the mouse being released over the canvas'''

        # Check that the mouse was actually pressed on the canvas to begin with and this isn't a rouge mouse
        # release event that started somewhere else
        if self.pressed:

            # Upon release draw the rectangle as a solid rectangle
            self.pressed = False
            self.rect.set_linestyle('solid')

            # Check the mouse was released on the canvas, and if it wasn't then just leave the width and
            # height as the last values set by the motion event
            if event.xdata is not None and event.ydata is not None:
                self.x1 = event.xdata
                self.y1 = event.ydata

            # Set the width and height and origin of the bounding rectangle
            self.boundingRectWidth =  self.x1 - self.x0
            self.boundingRectHeight =  self.y1 - self.y0
            self.bouningRectOrigin = (self.x0, self.y0)

            # Draw the bounding rectangle
            self.rect.set_width(self.boundingRectWidth)
            self.rect.set_height(self.boundingRectHeight)
            self.rect.set_xy((self.x0, self.y0))
            self.canvas.draw()

            int_x0 = int(self.x0)
            int_x1 = int(self.x1)
            int_y0 = int(self.y0)
            int_y1 = int(self.y1)

            temp_array = np.zeros((np.abs(int_y1 - int_y0), np.abs(int_x1 - int_x0)), dtype = np.uint8)

            y = 0
            x = 0
            if (int_y1 > int_y0):
                if (int_x1 > int_x0):
                    for row in range(int_y0, int_y1):
                        x = 0
                        for col in range(int_x0, int_x1):
                            temp_array[y][x] = self.image[row][col]
                            x = x + 1

                        y = y + 1
                else:
                    for row in range(int_y0, int_y1):
                        x = 0
                        for col in range(int_x1, int_x0):
                            temp_array[y][x] = self.image[row][col]
                            x = x + 1

                        y = y + 1
            elif (int_y1 < int_y0):
                if (int_x1 > int_x0):
                    for row in range(int_y1, int_y0):
                        x = 0
                        for col in range(int_x0, int_x1):
                            temp_array[y][x] = self.image[row][col]
                            x = x + 1

                        y = y + 1
                else:
                    for row in range(int_y1, int_y0):
                        x = 0
                        for col in range(int_x1, int_x0):
                            temp_array[y][x] = self.image[row][col]
                            x = x + 1

                        y = y + 1

            #cv2.imshow('image', temp_array)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            plt.close()
            plt.figure()
            vals = temp_array.flatten()
            # calculate histogram
            counts, bins = np.histogram(vals, range(257))
            # plot histogram centered on values 0..255
            plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
            plt.xlim([-0.5, 255.5])
            plt.show()


    def _onMotion(self, event):
        '''Callback to handle the motion event created by the mouse moving over the canvas'''

        # If the mouse has been pressed draw an updated rectangle when the mouse is moved so
        # the user can see what the current selection is
        if self.pressed:

            # Check the mouse was released on the canvas, and if it wasn't then just leave the width and
            # height as the last values set by the motion event
            if event.xdata is not None and event.ydata is not None:
                self.x1 = event.xdata
                self.y1 = event.ydata

            # Set the width and height and draw the rectangle
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.canvas.draw()



    def setImage(self, pathToImage):
        '''Sets the background image of the canvas'''

        # Load the image into matplotlib and PIL
        self.image = cv2.imread(pathToImage, cv2.IMREAD_GRAYSCALE)
        image = self.image
        imPIL = Image.open(pathToImage)


        # Save the image's dimensions from PIL
        self.imageSize = imPIL.size

        # Add the image to the figure and redraw the canvas. Also ensure the aspect ratio of the image is retained.
        self.axes.imshow(image, cmap = "gray")
        self.canvas.draw()

class BandPassFilter:
    image = None
    filter = None
    cutoff1 = None
    cutoff2 = None
    order = None
    output = None

    def __init__(self, image, filter_name, cutoff1, cutoff2, order = 0):
        self.image = image
        if filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter
        elif filter_name == 'gaussian_BPF':
            self.filter = self.get_gaussian_BPF

        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.order = order


    def get_gaussian_BPF(self, shape, cutoff1, cutoff2):
        height = shape[0]
        width = shape[1]
        gaussianBPF = np.zeros((height, width))
        gaussianImageL = self.get_gaussian_low_pass_filter(shape, cutoff1)
        gaussianImageH = self.get_gaussian_high_pass_filter(shape, cutoff2)
        for u in range(height):
            for v in range(width):
                gaussianBPF[u, v] = gaussianImageL[u, v] * gaussianImageH[u, v]

        return gaussianBPF

    def get_gaussian_low_pass_filter(self, shape, cutoff1):
        height = shape[0]
        width = shape[1]
        m = height / 2
        n = width / 2
        gaussianImageL = np.zeros((height, width))
        for u in range(height):
            for v in range(width):
                dist = self.getDist(u, v, m, n)
                gaussianImageL[u,v] = math.pow(math.e, -math.pow(dist,2)/(2*math.pow(cutoff1,2)))
        return gaussianImageL

    def get_gaussian_high_pass_filter(self, shape, cutoff2):
        #Hint: May be one can use the low pass filter function to get a high pass mask
        height = shape[0]
        width = shape[1]
        gaussianImageL = self.get_gaussian_low_pass_filter(shape, cutoff2)
        gaussianImageH = np.zeros((height, width))
        for u in range(height):
            for v in range(width):
                gaussianImageH[u,v] = 1 - gaussianImageL[u,v]
        return gaussianImageH

    def post_process_image(self, image):
        imageMin = np.min(image)
        imageMax = np.max(image)
        newImage = np.uint8(255 / (imageMax - imageMin) * (image - imageMin) + .5)
        return newImage


    def filtering(self):


        # 1 and 2
        fftImage = np.fft.fft2(self.image)
        shiftImage = np.fft.fftshift(fftImage)
        # 3
        mask = self.filter(self.image.shape, self.cutoff1, self.cutoff2)
        # 4, 5 and 6
        filteredImage = shiftImage * mask
        inverseShiftImage = np.fft.ifftshift(filteredImage)
        inverseShiftImage = np.fft.ifft2(inverseShiftImage)
        # 7
        magnitudeDFT = np.log(np.abs(shiftImage))
        mask = np.log(np.abs(mask))
        mask = np.abs(mask)
        magnitudeDFT = magnitudeDFT*20
        #magnitudeDFT = np.log(cv2.magnitude(shiftImage[:,:,0], shiftImage[:,:,1]))
        #magnitudeDFT, angle = cv2.cartToPolar(np.real(filteredImage), np.imag(filteredImage))
        magnitudeImage = np.log(np.abs(inverseShiftImage))
        magnitudeFilteredfft = np.log(np.abs(inverseShiftImage))
        # 8
        #magnitudeDFT = self.post_process_image(magnitudeDFT)
        magnitudeFilteredfft = self.post_process_image(magnitudeFilteredfft)
        finalFilteredImage = self.post_process_image(magnitudeImage)

        return [finalFilteredImage, magnitudeDFT, mask]

        #mask = mask.astype(np.uint8)
        #shiftImage = shiftImage.astype(np.complex_)
        #fftImage = fftImage.astype(np.uint8)
        #return [finalFilteredImage, shiftImage, magnitudeImage]


    def getDist(self, u, v, m, n):
        a = math.pow((u-m), 2)
        b = math.pow((v-n), 2)
        return math.sqrt(a+b)

class orderstatistic_filter:
    image = None
    filter = None
    #filter = 3x3

    #define constructor with parameters: image, filter_name
    def __init__(self, image, filter_name):
        self.image = image
        if filter_name == 'minimum':
            self.filter = self.get_min_filter
        elif filter_name == 'maximum':
            self.filter = self.get_max_filter
        elif filter_name == 'median':
            self.filter = self.get_median_filter
        elif filter_name == 'mean':
            self.filter = self.get_alpha_trimmed_mean_filter

    def get_min_filter(self, image):
        filter_h = 3
        filter_w = 3
        pad_h = int((filter_h - 1) / 2)
        pad_w = int((filter_w - 1) / 2)
        pad_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
        height, width = image.shape
        new_img = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                temp_img = pad_img[i:(i+filter_h), j:(j+filter_w)]
                # use min to find minimum value
                new_img[i, j] = np.min(temp_img)

        return new_img

    def get_max_filter(self, image):
        filter_h = 3
        filter_w = 3
        pad_h = int((filter_h - 1) / 2)
        pad_w = int((filter_w - 1) / 2)
        pad_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
        height, width = image.shape
        new_img = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                temp = pad_img[i:(i + filter_h), j:(j + filter_w)]
                # use max to find maximum value
                new_img[i, j] = np.max(temp)

        return new_img

    def get_median_filter(self, image):
        filter_h = 3
        filter_w = 3
        pad_h = int((filter_h - 1) / 2)
        pad_w = int((filter_w - 1) / 2)
        pad_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
        height, width = image.shape
        new_img = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                temp = pad_img[i:(i + filter_h), j:(j + filter_w)]
                # use np.median to find the median value
                new_img[i, j] = np.median(temp)

        return new_img

    def get_alpha_trimmed_mean_filter(self, image):
        filter_h = 3
        filter_w = 3
        pad_h = int((filter_h - 1) / 2)
        pad_w = int((filter_w - 1) / 2)
        pad_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0)
        height, width = image.shape
        new_img = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                temp = pad_img[i:(i + filter_h), j:(j + filter_w)]
                # to get alpha trimmed mean we need to first sort the values that were picked by the filter and then delete the first and last value then take the mean
                np.sort(temp)
                for x in [8,0]:
                    temp = np.delete(temp, x)

                new_img[i,j] = np.mean(temp)

        return new_img



    def filtering(self):
        if self.filter == self.get_min_filter:
            filtered_image = self.get_min_filter(self.image)
        elif self.filter == self.get_max_filter:
            filtered_image = self.get_max_filter(self.image)
        elif self.filter == self.get_median_filter:
            filtered_image = self.get_median_filter(self.image)
        elif self.filter == self.get_alpha_trimmed_mean_filter:
            filtered_image = self.get_alpha_trimmed_mean_filter(self.image)

        return filtered_image

class mean_filter:
	image=None
	filter=None
	order=None
	#define constructor with parameter: image, filter_name, filter window height AND WIDTH, order(contraharmonic only)
	def __init__(self,image,mean_filter_name,order=0):
		self.image=image
		if mean_filter_name=='arithmetic':
			self.filter=self.get_arithmetic_mean_filter
		elif mean_filter_name=='geometric':
			self.filter=self.get_geometric_mean_filter
		elif mean_filter_name=='harmonic':
			self.filter=self.get_harmonic_mean_filter
		elif mean_filter_name=='contraharmonic':
			self.filter=self.get_contraharmonic_mean_filter
		self.order=order

	def get_arithmetic_mean_filter(self,image):
		#apply zero padding, default window size 3x3
		#reference: https://stackoverflow.com/questions/44145948/numpy-padding-array-with-zeros
		#			https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
		filter_h=3
		filter_w=3
		pad_h=int((filter_h-1)/2)
		pad_w=int((filter_w-1)/2)
		padded_image=np.pad(image,((pad_h,pad_h),(pad_w,pad_w)),'constant',constant_values=0)
		#get width&height
		height,width=image.shape
		filtered_image=np.zeros((height,width))
		#self note: if size of image on GUI is diff, need to adjust***
		for row in range(height):
			for col in range(width):
				#get window of small part of image
				temp_image=padded_image[row:(row+filter_h),col:(col+filter_w)]
				#use np sum to get summation of all pixels and get mean of all pixels
				filtered_image[row,col]=1/(filter_h*filter_w)*np.sum(temp_image)

		return filtered_image

	def get_geometric_mean_filter(self,image):
		"""geometric mean filter achieves smoothing
		comparable to the arithmetic mean filter, but it tends to lose
		less image detail in the process"""
		#https://en.wiktionary.org/wiki/%CE%A0
		#apply zero padding
		filter_h=3
		filter_w=3
		pad_h=int((filter_h-1)/2)
		pad_w=int((filter_w-1)/2)
		padded_image=np.pad(image,((pad_h,pad_h),(pad_w,pad_w)),'constant',constant_values=0)
		#get width&height
		height,width=image.shape
		filtered_image=np.zeros((height,width))
		for row in range(height):
			for col in range(width):
				#get window of small part of image
				temp_image=padded_image[row:(row+filter_h),col:(col+filter_w)]
				# get Product over a set of terms: (Pi? Π?)
				product=1.0
				for p_h in range(filter_h):
					for p_w in range(filter_w):
						product=temp_image[p_h,p_w]*product
				filtered_image[row,col]=np.power(product,1/(filter_h*filter_w))

		return filtered_image

	def get_harmonic_mean_filter(self,image):
		"""
			Works well for SALT noise & Gaussian noise, fails for PEPPER noise
		"""
		#apply zero padding
		filter_h=3
		filter_w=3
		pad_h=int((filter_h-1)/2)
		pad_w=int((filter_w-1)/2)
		padded_image=np.pad(image,((pad_h,pad_h),(pad_w,pad_w)),'constant',constant_values=0)
		#get width&height
		height,width=image.shape
		filtered_image=np.zeros((height,width))
		for row in range(height):
			for col in range(width):
				#get window of small part of image
				temp_image=padded_image[row:(row+filter_h),col:(col+filter_w)]
				#mn-->filter window_h&w
				#f(x,y)=mn/ summation of (1/g(s,t)), g(s,t) is denominator
				# so, we only care about positive g(s,t) values.
				#get summation of 1/g(s,t)
				sum_g=0
				for s_h in range(filter_h):
					for s_w in range(filter_w):
						#only care about positive values
						if temp_image[s_h,s_w]!=0:
							sum_g+=1/temp_image[s_h,s_w]

				filtered_image[row,col]=filter_h*filter_w/sum_g

		return filtered_image

	def get_contraharmonic_mean_filter(self,image,order):
		"""well suited for reducing the effects of SALT-and-PEPPER
		noise. Q>0 for pepper noise and Q<0 for salt noise. (Q: order of filter)

		"""
		#apply zero padding
		filter_h=3
		filter_w=3
		pad_h=int((filter_h-1)/2)
		pad_w=int((filter_w-1)/2)
		padded_image=np.pad(image,((pad_h,pad_h),(pad_w,pad_w)),'constant',constant_values=0)
		#get width&height
		height,width=image.shape
		filtered_image=np.zeros((height,width))
		for row in range(height):
			for col in range(width):
				#get window of small part of image
				temp_image=padded_image[row:(row+filter_h),col:(col+filter_w)]
				#f(x,y)=summation of g(s,t)^(Q+1)  /   summation of g(s,t)^(Q)
				#similar to harmonic: g(s,t)^(Q) is denominator cant be zero
				sum_g1=0
				sum_g2=0
				for s_h in range(filter_h):
					for s_w in range(filter_w):
						#get power of Q of each intensity and sum
						sum_g1+=np.power(temp_image[s_h,s_w], order+1)
						sum_g2+=np.power(temp_image[s_h,s_w], order)
				if sum_g2!=0:
					filtered_image[row,col]=sum_g1/sum_g2
				else:
					filtered_image[row,col]=0

		return filtered_image

	def filtering(self):
		if self.filter==self.get_arithmetic_mean_filter:
			filtered_image=self.get_arithmetic_mean_filter(self.image)
		elif self.filter==self.get_geometric_mean_filter:
			filtered_image=self.get_geometric_mean_filter(self.image)
		elif self.filter==self.get_harmonic_mean_filter:
			filtered_image=self.get_harmonic_mean_filter(self.image)
		elif self.filter==self.get_contraharmonic_mean_filter:
			filtered_image=self.get_contraharmonic_mean_filter(self.image,self.order)

		return filtered_image

class MyFrame3 ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 635,439 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

		bSizer16 = wx.BoxSizer( wx.HORIZONTAL )

		gSizer10 = wx.GridSizer( 0, 2, 0, 0 )

		self.m_bitmap6 = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer10.Add( self.m_bitmap6, 0, wx.ALL, 5 )

		self.m_bitmap7 = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer10.Add( self.m_bitmap7, 0, wx.ALL, 5 )

		self.m_bitmap8 = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer10.Add( self.m_bitmap8, 0, wx.ALL, 5 )

		self.m_bitmap9 = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer10.Add( self.m_bitmap9, 0, wx.ALL, 5 )


		bSizer16.Add( gSizer10, 1, wx.EXPAND, 5 )

		bSizer22 = wx.BoxSizer( wx.VERTICAL )

		self.m_radioBtn38 = wx.RadioButton( self, wx.ID_ANY, u"Original Image", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer22.Add( self.m_radioBtn38, 0, wx.ALL, 5 )

		m_choice2Choices = [ u"Select Mean Filter", u"Arithmetic Mean Filter", u"Geometric Mean Filter", u"Harmonic Mean Filter", u"Contraharmonic Mean Filter"]
		self.m_choice2 = wx.Choice( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_choice2Choices, 0 )
		self.m_choice2.SetSelection( 0 )
		bSizer22.Add( self.m_choice2, 0, wx.ALL, 5 )

		m_choice3Choices = [ u"Select Order Statistic Filter", u"Minimum Order Statistic Filter", u"Maximum Order Statistic Filter", u"Median Order Statistic Filter", u"Trimmed Mean Order Statistic Filter" ]
		self.m_choice3 = wx.Choice( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_choice3Choices, 0 )
		self.m_choice3.SetSelection( 0 )
		bSizer22.Add( self.m_choice3, 0, wx.ALL, 5 )

		m_choice4Choices = [ u"Select Adaptive Filter", u"Adaptive Reduction Filter", u"Adaptive Median Filter" ]
		self.m_choice4 = wx.Choice( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_choice4Choices, 0 )
		self.m_choice4.SetSelection( 0 )
		bSizer22.Add( self.m_choice4, 0, wx.ALL, 5 )

		self.m_radioBtn32 = wx.RadioButton( self, wx.ID_ANY, u"Band Reject Filter", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer22.Add( self.m_radioBtn32, 0, wx.ALL, 5 )

		self.m_radioBtn33 = wx.RadioButton( self, wx.ID_ANY, u"Notch Filter", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer22.Add( self.m_radioBtn33, 0, wx.ALL, 5 )

		bSizer23 = wx.BoxSizer( wx.HORIZONTAL )

		bSizer22.Add( bSizer23, 0, wx.EXPAND, 5 )

		self.m_radioBtn34 = wx.RadioButton( self, wx.ID_ANY, u"Band Pass Filter", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer22.Add( self.m_radioBtn34, 0, wx.ALL, 5 )

		self.m_button198 = wx.Button( self, wx.ID_ANY, u"Noise Sampling", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer22.Add( self.m_button198, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_button199 = wx.Button( self, wx.ID_ANY, u"Pick Image", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer22.Add( self.m_button199, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_button200 = wx.Button( self, wx.ID_ANY, u"Save Image", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer22.Add( self.m_button200, 0, wx.ALL|wx.EXPAND, 5 )


		bSizer16.Add( bSizer22, 0, wx.EXPAND, 5 )


		self.SetSizer( bSizer16 )
		self.Layout()

		self.Centre( wx.BOTH )

	def getBound(self, event):
		self.m_bitmap8.SetBitmap(wx.NullBitmap)
		self.Refresh()
		self.m_bitmap9.SetBitmap(wx.NullBitmap)
		self.Refresh()
		lBound = self.ask(message = "Insert your lower bound")
		lBound = int(lBound)
		hBound = self.ask(message = "Insert your upper bound")
		hBound = int(hBound)
		self.onButtonNotchFilter(lBound, hBound)

	def onButtonBandPassFilter(self, event):
		self.m_bitmap9.SetBitmap(wx.NullBitmap)
		self.m_bitmap8.SetBitmap(wx.NullBitmap)
		self.Refresh()
		lCutoff = self.ask(message = "Insert your lower cutoff")
		lCutoff = int(lCutoff)
		hCutoff = self.ask(message = "Insert your upper cutoff")
		hCutoff = int(hCutoff)

		Filter_obj = BandPassFilter(self.orig_image, "gaussian_BPF", lCutoff, hCutoff)
		output = Filter_obj.filtering()

		self.image = output
		cv2.imwrite("test.png", output[0])
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap7.SetBitmap(img)
		os.remove("test.png")

		cv2.imwrite("test.png", output[2])
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap8.SetBitmap(img)
		os.remove("test.png")

		cv2.imwrite("test.png", output[1])
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap9.SetBitmap(img)
		os.remove("test.png")

	def onButtonAdaptiveFilter(self, event):
		self.m_bitmap9.SetBitmap(wx.NullBitmap)
		self.m_bitmap8.SetBitmap(wx.NullBitmap)
		self.Refresh()
		if (self.m_choice4.GetStringSelection() == "Adaptive Reduction Filter"):
			Filter_obj = adaptive_filter(self.orig_image, "reduction")
			output = Filter_obj.filtering()
		elif (self.m_choice4.GetStringSelection() == "Adaptive Median Filter"):
			Filter_obj = adaptive_filter(self.orig_image, "median")
			output = Filter_obj.filtering()
		else:
			self.m_bitmap7.SetBitmap(wx.NullBitmap)
			self.Refresh()
			return

		self.image = output
		cv2.imwrite("test.png", output)
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap7.SetBitmap(img)
		os.remove("test.png")

	def onButtonOrderStatisticFilter(self, event):
		self.m_bitmap9.SetBitmap(wx.NullBitmap)
		self.m_bitmap8.SetBitmap(wx.NullBitmap)
		self.Refresh()
		if (self.m_choice3.GetStringSelection() == "Minimum Order Statistic Filter"):
			Filter_obj = orderstatistic_filter(self.orig_image, "minimum")
			output = Filter_obj.filtering()
		elif (self.m_choice3.GetStringSelection() == "Maximum Order Statistic Filter"):
			Filter_obj = orderstatistic_filter(self.orig_image, "maximum")
			output = Filter_obj.filtering()
		elif (self.m_choice3.GetStringSelection() == "Median Order Statistic Filter"):
			Filter_obj = orderstatistic_filter(self.orig_image, "median")
			output = Filter_obj.filtering()
		elif (self.m_choice3.GetStringSelection() == "Trimmed Mean Order Statistic Filter"):
			Filter_obj = orderstatistic_filter(self.orig_image, "mean")
			output = Filter_obj.filtering()
		else:
			self.m_bitmap7.SetBitmap(wx.NullBitmap)
			self.Refresh()
			return

		self.image = output
		cv2.imwrite("test.png", output)
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap7.SetBitmap(img)
		os.remove("test.png")

	def post_process_image(self, image):
		image = np.array(image)
		minimumValue = np.amin(image)
		maximumValue = np.amax(image)

		P = (255/(maximumValue-minimumValue))
		outputImage = image
		cols, rows = image.shape

		for j in range(rows-1):
			for i in range(cols-1):
				outputImage[i, j] = P*(image[i, j]-minimumValue)

		return outputImage

	def ask(parent=None, message='', default_value=''):
		dlg = wx.TextEntryDialog(parent, message, default_value)
		dlg.ShowModal()
		result = dlg.GetValue()
		dlg.Destroy()
		return result

	def onButtonMeanFilter(self, event):
		self.m_bitmap9.SetBitmap(wx.NullBitmap)
		self.m_bitmap8.SetBitmap(wx.NullBitmap)
		self.Refresh()
		if (self.m_choice2.GetStringSelection() == "Arithmetic Mean Filter"):
			Filter_obj = mean_filter(self.orig_image, "arithmetic")
			output = Filter_obj.filtering()
		elif (self.m_choice2.GetStringSelection() == "Geometric Mean Filter"):
			Filter_obj = mean_filter(self.orig_image, "geometric")
			output = Filter_obj.filtering()
		elif (self.m_choice2.GetStringSelection() == "Harmonic Mean Filter"):
			Filter_obj = mean_filter(self.orig_image, "harmonic")
			output = Filter_obj.filtering()
		elif (self.m_choice2.GetStringSelection() == "Contraharmonic Mean Filter"):
			order = self.ask(message = "Insert your order")
			order = int(order)
			Filter_obj = mean_filter(self.orig_image, "contraharmonic", order)
			output = Filter_obj.filtering()
		else:
			self.m_bitmap7.SetBitmap(wx.NullBitmap)
			self.Refresh()
			return

		self.image = output
		cv2.imwrite("test.png", output)
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap7.SetBitmap(img)
		os.remove("test.png")

	def onButtonOpenFile(self, event):
		f = wx.FileDialog(self, "Open image file", wildcard="Image files (*.png)|*.png", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

		if f.ShowModal() == wx.ID_CANCEL:
			return

		# Proceed loading the file chosen by the user
		pathname = f.GetPath()
		try:
			with open(pathname, 'r') as file:
				self.m_bitmap6.SetBitmap(wx.NullBitmap)
				self.Refresh()
				self.image = cv2.imread(pathname, cv2.IMREAD_GRAYSCALE)
				self.orig_image = self.image
				img = wx.Image(pathname, wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
				self.m_bitmap6.SetBitmap(img)
				self.m_bitmap9.SetBitmap(wx.NullBitmap)
				self.m_bitmap8.SetBitmap(wx.NullBitmap)
				self.m_bitmap7.SetBitmap(wx.NullBitmap)
				self.Refresh()
		except IOError:
			wx.LogError("Cannot open file")

	def onButtonSaveImage(self, event):
		f = wx.FileDialog(self, "Save image file", wildcard="Image files (*.png)|*.png", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

		if f.ShowModal() == wx.ID_CANCEL:
			return

		pathname = f.GetPath()
		try:
			with open(pathname, 'w') as f:
				cv2.imwrite(pathname, self.image)
		except:
			print("Save failed")

	def onButtonNotchFilter(self, lBound, hBound):
		self.m_bitmap9.SetBitmap(wx.NullBitmap)
		self.m_bitmap8.SetBitmap(wx.NullBitmap)
		self.Refresh()
		img = self.orig_image
		img = np.array(img)

		dft_img = np.fft.fftshift(np.fft.fft2(img))
		mag_dft_img = np.log(np.abs(dft_img))

		temp_img = mag_dft_img*20
		cv2.imwrite("test.png", temp_img)
		temp_img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap9.SetBitmap(temp_img)
		os.remove("test.png")

		mag_dft_img = mag_dft_img.astype(np.uint8)

		mag_dft_img = self.post_process_image(mag_dft_img)

		cols, rows = img.shape
		bigcircle = np.zeros((cols, rows))

		bigcircle = self.get_ideal_high_pass_filter( bigcircle.shape, hBound)

		cols, rows = img.shape
		smallcircle = np.zeros((cols, rows))

		smallcircle = self.get_ideal_low_pass_filter( smallcircle.shape, lBound)

		final_filter = smallcircle + bigcircle

		temp = final_filter
		temp = temp*100
		cv2.imwrite("test.png", temp)
		temp = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap8.SetBitmap(temp)
		os.remove("test.png")

		filtered_image = final_filter * dft_img

		filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_image))

		filtered_image = filtered_image.astype(np.uint8)

		final_filter = self.post_process_image(final_filter)

		self.image = filtered_image
		cv2.imwrite("test.png", filtered_image)
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap7.SetBitmap(img)
		os.remove("test.png")

	def get_ideal_high_pass_filter(self, shape, cutoff):
	        cols, rows = shape

	        final_filter = self.get_ideal_low_pass_filter(shape, cutoff)

	        for j in range(rows-1):
	            for i in range(cols-1):
	                final_filter[i,j] = 1-final_filter[i,j]

	        return final_filter

	def get_ideal_low_pass_filter(self, shape, cutoff):
	        cols, rows = shape

	        final_filter = np.zeros((cols, rows))

	        cenRows = rows/2
	        cenCols = cols/2

	        for j in range(rows):
	            for i in range(cols):
	                y = np.square(j-cenRows)
	                x = np.square(i-cenCols)

	                if np.sqrt(x+y) <= cutoff:
	                    final_filter[i, j] = 1

	        return final_filter

	def onButtonOriginal(self, event):
		self.m_bitmap7.SetBitmap(wx.NullBitmap)
		self.m_bitmap8.SetBitmap(wx.NullBitmap)
		self.m_bitmap9.SetBitmap(wx.NullBitmap)
		self.Refresh()

		self.image = self.orig_image

	def onButtonNoiseSample(self, event):
		image_frame = wx.Frame(None, title="image")
		panel = RectangleSelectImagePanel(image_frame)
		cv2.imwrite("test.png", self.orig_image)
		panel.setImage("test.png")
		os.remove("test.png")
		image_frame.Show()

	def onButtonBandRejectFilter(self, event):
		self.m_bitmap9.SetBitmap(wx.NullBitmap)
		self.m_bitmap8.SetBitmap(wx.NullBitmap)
		self.Refresh()
		lCutoff = self.ask(message = "Insert your lower cutoff")
		lCutoff = int(lCutoff)
		hCutoff = self.ask(message = "Insert your upper cutoff")
		hCutoff = int(hCutoff)

		Filter_obj = BandRejectFilter(self.orig_image, "gaussian_BRF", lCutoff, hCutoff)
		output = Filter_obj.filtering()

		self.image = output
		cv2.imwrite("test.png", output[0])
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap7.SetBitmap(img)
		os.remove("test.png")

		cv2.imwrite("test.png", output[2])
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap8.SetBitmap(img)
		os.remove("test.png")

		cv2.imwrite("test.png", output[1])
		img = wx.Image("test.png", wx.BITMAP_TYPE_ANY).ConvertToGreyscale().ConvertToBitmap()
		self.m_bitmap9.SetBitmap(img)
		os.remove("test.png")

	def __del__( self ):
		pass

app = wx.App()
frame = MyFrame3(None)
frame.Show()
frame.m_radioBtn33.Bind(wx.EVT_RADIOBUTTON, frame.getBound)
frame.m_button199.Bind(wx.EVT_BUTTON, frame.onButtonOpenFile)
frame.m_button200.Bind(wx.EVT_BUTTON, frame.onButtonSaveImage)
frame.m_choice2.Bind(wx.EVT_CHOICE, frame.onButtonMeanFilter)
frame.m_choice3.Bind(wx.EVT_CHOICE, frame.onButtonOrderStatisticFilter)
frame.m_choice4.Bind(wx.EVT_CHOICE, frame.onButtonAdaptiveFilter)
frame.m_radioBtn34.Bind(wx.EVT_RADIOBUTTON, frame.onButtonBandPassFilter)
frame.m_radioBtn38.Bind(wx.EVT_RADIOBUTTON, frame.onButtonOriginal)
frame.m_button198.Bind(wx.EVT_BUTTON, frame.onButtonNoiseSample)
frame.m_radioBtn32.Bind(wx.EVT_RADIOBUTTON, frame.onButtonBandRejectFilter)
app.MainLoop()
