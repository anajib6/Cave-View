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

		# if the match is None, then there aren't eough matched
		# keypoints to create a panorama
		if M is None:
			print 'Not Enough Matches Found'
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together

		(matches, H, status) = M
		warpedHeight, warpedWidth, _ = warpedImage.shape
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
		# BLENDING CAN BE DONE HERE
		# result[0:sourceImage.shape[0], 0:sourceImage.shape[1]] = sourceImage
		cond = np.where(sourceImage > [0, 0, 0]) # note: if imageA is a merged image, it will have black padded to sides
		result[cond] = sourceImage[cond]
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
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
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
args = vars(ap.parse_args())
image_name = args['image']
match = args['match']
stitcher = Stitcher()
sourceImage = None
for i, imagePath in enumerate(paths.list_images('images/{}'.format(image_name))):
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

	matched_result = stitcher.stitch([sourceImage, warpImage], showMatches=match)
	(result, vis) = matched_result
	cv2.imshow('stitched', result)
	if match:
		cv2.imshow("Keypoint Matches", vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	sourceImage = result
