import numpy as np
import imutils
import cv2

# Code adapted from https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(sourceImage, warpedImage) = images
		(kpsA, featuresA) = self.detectAndDescribe(warpedImage)
		(kpsB, featuresB) = self.detectAndDescribe(sourceImage)

		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)

		# if the match is None or Homograph Matrix is None, then there aren't eough matched
		# keypoints to create a panorama
		if M is None or M[1] is None:
			print 'Not Enough Matches Found'
			return None
		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		warpedHeight, warpedWidth, _ = warpedImage.shape
		# print "warpedHeight",warpedHeight
		# print "warpedWidth", warpedWidth
		# print type(H)
		warpTopRight = np.matmul(H, np.array([warpedWidth, 0, 1]))
		warpTopRight = warpTopRight / warpTopRight[2] # normalization
		warpBotRight = np.matmul(H, np.array([warpedWidth, warpedHeight, 1]))
		warpBotRight = warpBotRight / warpBotRight[2] # normalization
		warpShorterBorderWidth = min(int(warpTopRight[0]), int(warpBotRight[0]))
		finalImageWidth = max(sourceImage.shape[1], warpShorterBorderWidth)

		warpTopLeft = np.matmul(H, np.array([warpedWidth, 0, 1]))
		warpTopLeft = warpTopLeft / warpTopLeft[2] # normalization
		warpBotLeft = np.matmul(H, np.array([warpedWidth, warpedHeight, 1]))
		warpBotLeft = warpBotLeft / warpBotLeft[2] # normalization
		finalImageHeight = max(sourceImage.shape[0], int(warpBotRight[0]), int(warpBotLeft[0]))
		result = cv2.warpPerspective(
			warpedImage, H,
			(finalImageWidth, finalImageHeight)
		)

		# we merge sourceImage into final image

		# result[0:sourceImage.shape[0], 0:sourceImage.shape[1]] = sourceImage
		cond = np.where(sourceImage > [0, 0, 0]) # note: if imageA is a merged image, it will have black padded to sides

		res_only = np.where(result > [0,0,0])
		null_res = np.where(result == [0,0,0])

		res_top = np.zeros(result.shape,dtype=int)
		t_source_top = np.zeros(result.shape, dtype=int) 
		source_top = np.zeros(result.shape, dtype=int) 

		t_source_top[cond] = sourceImage[cond]
		null_src = np.where(t_source_top == [0,0,0])
		res_top[null_src] = result[null_src]

		# create array size of full
		combined = res_top + source_top
		null_area = np.where(combined > [0,0,0])
		binary = np.zeros(result.shape, dtype=np.uint8)
		binary[null_area] = 255

		# null_area_2 = (null_area[0] + sourceImage.shape[0]/10, null_area[1] - )
		# import matplotlib.pyplot as plt

		# offset_y = sourceImage.shape[0]/25
		# offset_x = sourceImage.shape[1]/25
		offset_y = 5
		offset_x = 5
		min0 = max(np.min(null_area[0] + offset_y),0,np.min(res_only[0])+offset_y)
		max0 = min(np.max(null_area[0] - offset_y), binary.shape[0], np.max(res_only[0])-offset_y)
		min1 = max(np.min(null_area[1] + offset_x),0, np.min(res_only[1])+offset_x)
		max1 = min(np.max(null_area[1] - offset_x), binary.shape[1], np.max(res_only[1])-offset_x)
		binary[min0:max0, min1:max1] = 255

		# binary = cv2.cvtColor(binary, cv2.COLOR_RGB2GRAY)
		binary = cv2.blur(binary,(offset_y,offset_x))
		binary = binary/255.0
		source_top[null_res] = t_source_top[null_res]
		# scale points by averages alpha transform
		out = np.multiply(source_top + result, binary) + np.multiply(res_top+t_source_top,1-binary)
		out = np.array(out , dtype=np.uint8)
	# fig, ax = plt.subplots(3,3)
		# ax[0,0].imshow(binary)
		# ax[0,1].imshow(out)
		# ax[0,2].imshow(res_top)
		# ax[1,0].imshow(source_top)
		# ax[1,1].imshow(t_source_top)
		# ax[1,2].imshow(result)
		# ax[2,0].imshow(warpedImage)
		# ax[2,1].imshow(sourceImage)
		# # ax[2,2].imshow(result)
		# plt.show()
		# plt.close()

		result = out
		# result[cond] = sourceImage[cond]
		row, col, _ = result.shape
		#  naiive clipping
		for y in range(row-1, -1, -1):
			trim = False
			for x in range(col):
				if not np.array_equal(result[y, x], [0, 0, 0]):
					trim = True
			if trim:
				result = result[:y]
				break

		if showMatches:
			vis = self.drawMatches(warpedImage, sourceImage, kpsA, kpsB, matches,
				status)

			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)

		# return the stitched image
		return (result, None)

	def detectAndDescribe(self, image):
			# convert the image to grayscale
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# check to see if we are using OpenCV 3.X
			if self.isv3:
				# detect and extract features from the image
				descriptor = cv2.xfeatures2d.SIFT_create()
				(kps, features) = descriptor.detectAndCompute(image, None)

			# otherwise, we are using OpenCV 2.4.X
			else:
				# detect keypoints in the image
				detector = cv2.FeatureDetector_create("SIFT")
				kps = detector.detect(gray)

				# extract features from the image
				extractor = cv2.DescriptorExtractor_create("SIFT")
				(kps, features) = extractor.compute(gray, kps)

			# convert the keypoints from KeyPoint objects to NumPy
			# arrays
			kps = np.float32([kp.pt for kp in kps])

			# return a tuple of keypoints and features
			return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.FM_RANSAC,
				reprojThresh)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)
		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
		colors = [(0, 255,0), (0, 0, 255), (255, 0, 0)]
		# loop over the matches
		index = 0
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, colors[index % 3], 1)
				index += 1

		# return the visualization
		return vis

	def cylindricalWarp(self, image):
		import math
		height, width, rgb = image.shape
		yc = int(height/2)
		xc = int(width/2)
		warpedImage = np.zeros((2*height, 2*width, rgb), dtype=np.uint8)
		focalLength = width * (5.4 / 3.2) # EXIF DATA AND GOOGLE AND MAGIC NUMBER
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


import argparse
from imutils import paths
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True)
ap.add_argument('-m', '--match', action='store_true', default=False)
ap.add_argument('-s', '--split', action='store_true', default=False)
args = vars(ap.parse_args())
image_name = args['image']
match = args['match']
split = args['split']
stitcher = Stitcher()
sourceImage = None
# for i, imagePath in enumerate(sorted(list(paths.list_images('images/{}'.format(image_name))))):
	# print imagePath
for i, imagePath in enumerate(sorted(list(paths.list_images('images/{}'.format(image_name))))):
# for i, imagePath in enumerate(paths.list_images('images/{}'.format(image_name))):
	if i == 0:
		sourceImage = cv2.imread(imagePath)
		sourceImage = imutils.resize(sourceImage, width=150)
		sourceImage = stitcher.cylindricalWarp(sourceImage)
		continue

	# load the two images and resize them to have a width of 400 pixels
	# (for faster processing)
	warpImage = cv2.imread(imagePath)
	warpImage = imutils.resize(warpImage, width=150)
	warpImage = stitcher.cylindricalWarp(warpImage)
	if split and i > 4:
		leftSourceImage = sourceImage.copy()[:,:sourceImage.shape[1]/2]
		rightSourceImage = sourceImage.copy()[:,sourceImage.shape[1]/2:]
		matched_result = stitcher.stitch([rightSourceImage, warpImage], showMatches=match)
	else:
		matched_result = stitcher.stitch([sourceImage, warpImage], showMatches=match)

# if no matches found, stop the loop
	if matched_result is None:
		cv2.destroyAllWindows()
		break
	(result, vis) = matched_result
	if split and i > 4:
		height = result.shape[0] if result.shape[0] > leftSourceImage.shape[0] else leftSourceImage.shape[0]
		width = result.shape[1] + leftSourceImage.shape[1]
		merge = np.zeros((height,width,3)).astype("uint8")
		merge[:leftSourceImage.shape[0], :leftSourceImage.shape[1]] = leftSourceImage
		merge[:result.shape[0], leftSourceImage.shape[1]:] = result
		result = merge

	cv2.imshow('stitched', result)
	if match:
		cv2.imshow("Keypoint Matches", vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	sourceImage = result
