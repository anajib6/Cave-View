import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageTransform:
	@staticmethod
	def get_final_image_dimensions(H, sourceImage, warpedImage):
		warpedHeight, warpedWidth, _ = warpedImage.shape
		finalImageWidth = warpedWidth + sourceImage.shape[1]

		warpBotRight = np.matmul(H, np.array([warpedWidth, warpedHeight, 1]))
		warpBotRight = warpBotRight / warpBotRight[2] # normalization
		warpBotLeft = np.matmul(H, np.array([warpedWidth, warpedHeight, 1]))
		warpBotLeft = warpBotLeft / warpBotLeft[2] # normalization
		finalImageHeight = max(sourceImage.shape[0], int(warpBotRight[0]), int(warpBotLeft[0]))

		return finalImageWidth, finalImageHeight

	@staticmethod
	def crop_image(img, tol=50):
		# img is image data
		# tol  is tolerance
		gimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		mask = gimg > tol
		return img[np.ix_(mask.any(1),mask.any(0))]

	@staticmethod
	def cylindrical_warp(image, i=0):
		height, width, rgb = image.shape
		yc = int(height/2)
		xc = int(width/2)
		warpedImage = np.zeros((2*height, 2*width, rgb), dtype=np.uint8)
		# focalLength = width * (5.4 / (3.2+0.01*i)) # EXIF DATA AND GOOGLE AND MAGIC NUMBER
		focalLength = width * (4 + 0.1 * i/ (1)) # EXIF DATA AND GOOGLE AND MAGIC NUMBER
		maxY = -float('inf')
		maxX = -float('inf')
		minY = float('inf')
		minX = float('inf')
		for y in range(height):
			for x in range(width):
				xprime = focalLength * math.atan((x-xc)/focalLength)
				yprime = focalLength * (y-yc)/math.sqrt((x-xc)**2 + focalLength**2)
				xprime += width
				yprime += height
				maxY = max(yprime, maxY)
				maxX = max(xprime, maxX)
				minY = min(yprime, minY)
				minX = min(xprime, minX)
				if xprime < 0 or xprime >= 2*width or yprime < 0 or yprime >= 2*height:
					continue
				warpedImage[int(yprime), int(xprime)] = image[y, x]
		maxY = int(maxY)
		maxX = int(maxX)
		minY = int(minY)
		minX = int(minX)
		warpedImage = warpedImage[minY:maxY, minX:maxX]
		return warpedImage

	@staticmethod
	def defisheye(image): # doesnt' work well
		import math
		height, width, rgb = image.shape
		center = (height/2, width/2)
		copy = np.zeros((height, width, rgb), dtype='uint8')
		for i in range(height):
			for j in range(width):
				di = center[0] - i
				dj = center[1] - j
				rd = (i - center[0]) ** 2 + (j - center[1]) ** 2
				rd = math.sqrt(rd)
				if rd == 0:
					print i, j
					copy[i, j] = image[i, j]
					continue
				f = width * 7000
				ru = math.tan(rd/f) * f
				scale = ru/rd
				# print rd, ru, di, dj, scale
				new_i = int(round(center[0] - di * scale))
				new_j = int(round(center[1] - dj * scale))
				if new_i < 0 or new_i >= height or new_j < 0 or new_j >= width:
					# print new_i, new_j, scale
					continue

				copy[new_i, new_j] = image[i, j]
		return ImageTransform.crop_image(copy)



	@staticmethod
	def straighten(image, k=1):
		image = cv2.copyMakeBorder(image,200,200,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
		plt.imshow(image)
		plt.show()
		gimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		height, width = gimage.shape
		top_right = -1

		for i in range(height / 2):
			for j in range(width, 0 , -1):
				if gimage[i, j-1] > 0:
					top_right = [j-1, i]
					break
			if top_right != -1:
				break
		br = False
		for j in range(width, 0 , -1):
			for i in range(height / 2):
				if gimage[i, j-1] > 0:
					if i - top_right[1] < 10 * k:
						top_right = [j-1, i]
						br = True
						break
			if br:
				break

		bot_right = -1
		for i in range(height, height/2, -1):
			for j in range(width * 3/4, width):
				if gimage[i-1, j-1] > 0:
					bot_right = [j-1, i-1]
					break
			if bot_right != -1:
				break

		top = -1
		bottom = -1
		for i in range(height/2):
			for j in range(width):
				if gimage[i, j] > 0:
					top = i
					break
			if top != -1:
				break

		bottom = -1
		for i in range(height, height/2, -1):
			if gimage[i-1, 0] > 0:
				bottom = i-1
				break

		width = min(bot_right[0], top_right[0])
		top = 200
		#---- 4 corner points of the bounding box
		pts_src = np.array([[0.0, top], top_right, [0.0, bottom], bot_right])
		print pts_src

		#---- 4 corner points of the black image you want to impose it on
		pts_dst = np.array([[0.0, top], [width, top], [0.0, bottom], [width, bottom]])

		#---- forming the black image of specific size
		im_dst = np.zeros((bottom, width, 3), np.uint8)

		#---- Framing the homography matrix
		h, status = cv2.findHomography(pts_src, pts_dst)
		#---- transforming the image bound in the rectangle to straighten
		im_out = cv2.warpPerspective(image, h, (im_dst.shape[1],im_dst.shape[0]))
		im_out = ImageTransform.crop_image(im_out, tol=10)
		magic = zip(*pts_src)
		magic2 = zip(*pts_dst)

		# for debug
		plt.imshow(gimage)
		plt.scatter(magic[0], magic[1])
		plt.scatter(magic2[0], magic2[1], c='r')
		plt.show()
		plt.imshow(im_out)
		plt.show()

		return im_out
