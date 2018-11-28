import cv2
import numpy as np

class ImageCorrection:
	@staticmethod
	def get_average_intensity(image):
		gimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		return np.average(gimg)

	@staticmethod
	def adjust_gamma(image, gamma=1.0):
		# build a lookup table mapping the pixel values [0, 255] to
		# their adjusted gamma values
		cv2.imshow('b', image)
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
		# apply gamma correction using the lookup table
		final = cv2.LUT(image, table)
		cv2.imshow('a', final)
		return final

	@staticmethod
	def clahe_illum(image):

		#-----Converting image to LAB Color model-----------------------------------
		lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
		cl = clahe.apply(l)
		limg = cv2.merge((cl,a,b))
		final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
		return final
