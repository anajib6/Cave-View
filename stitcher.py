import numpy as np
import imutils
import cv2
import math
import matplotlib.pyplot as plt
from transform import ImageTransform
from correction import ImageCorrection
# Code adapted from htts://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def stitch(self, images, ratio=0.7, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(sourceImage, warpedImage) = images
		# sourceIntensity = ImageCorrection.get_average_intensity(sourceImage)
		# warpIntensity = ImageCorrection.get_average_intensity(warpedImage)
		# gamma = sourceIntensity/warpIntensity # estimated brightness change
		# warpedImage = ImageCorrection.adjust_gamma(warpedImage, gamma)
		# print gamma, sourceIntensity, warpIntensity, ImageCorrection.get_average_intensity(warpedImage)
		(kpsA, featuresA) = self.detectAndDescribe(warpedImage)
		(kpsB, featuresB) = self.detectAndDescribe(sourceImage)
		# cv2.imshow('source', sourceImage)
		# cv2.imshow('pre-warped', warpedImage)
		# cv2.waitKey(0)
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
		finalImageWidth, finalImageHeight = ImageTransform.get_final_image_dimensions(H, sourceImage, warpedImage)
		result = cv2.warpPerspective(
			warpedImage, H,
			(finalImageWidth, finalImageHeight)
		)
		# we merge sourceImage into final image
		cv2.imshow('warped', result)
		cv2.waitKey(0)
		# result[0:sourceImage.shape[0], 0:sourceImage.shape[1]] = sourceImage

		sourceGrayImage = cv2.cvtColor(sourceImage, cv2.COLOR_RGB2GRAY)
		resGrayImage = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
		# cond = np.where(resGrayImage == 0) # note: if imageA is a merged image, it will have black padded to sides

		# result[0:sourceImage.shape[0], 0:sourceImage.shape[1]] = sourceImage
		cond = np.where(sourceGrayImage > 0) # note: if imageA is a merged image, it will have black padded to sides
		for w, h in zip(*cond):
			res = resGrayImage[w, h]
			src = sourceGrayImage[w, h]
			if src > res:
				result[w, h] = sourceImage[w, h]

		row, col, _ = result.shape
		result = ImageTransform.crop_image(result)

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

			image = ImageCorrection.clahe_illum(image)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# # cv2.waitKey(0)
			# cv2.destroyAllWindows()

			# check to see if we are using OpenCV 3.X
			if self.isv3:
				# detect and extract features from the image
				descriptor = cv2.xfeatures2d.SIFT_create()
				# descriptor = cv2.
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
		vis[0:hB, 0:wB] = imageB
		vis[0:hA, wB:] = imageA
		colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
		# loop over the matches
		index = 0
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]) + wB, int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, colors[index % 3], 1)
				index += 1

		# return the visualization
		return vis

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
image_width = 500
for i, imagePath in enumerate(sorted(list(paths.list_images('images/{}'.format(image_name))))):
	print imagePath
	if i == 0:
		sourceImage = cv2.imread(imagePath)
		sourceImage = imutils.resize(sourceImage, width=image_width)
		sourceImage = ImageTransform.cylindrical_warp(sourceImage, i)
		sourceImage = ImageTransform.crop_image(sourceImage)
		continue

	# load the two images and resize them to have a width of 400 pixels
	# (for faster processing)
	warpImage = cv2.imread(imagePath)
	warpImage = imutils.resize(warpImage, width=image_width)
	warpImage = ImageTransform.cylindrical_warp(warpImage, i)
	warpImage = ImageTransform.crop_image(warpImage)

	if split and i > 1:
		print 'split', i
		leftSourceImage = sourceImage.copy()[:,:sourceImage.shape[1]/2]
		rightSourceImage = sourceImage.copy()[:,sourceImage.shape[1]/2:]
		matched_result = stitcher.stitch([rightSourceImage, warpImage], showMatches=match)
	else:
		matched_result = stitcher.stitch([sourceImage, warpImage], showMatches=match)

# if no matches found, stop the loop
	if matched_result is None:
		cv2.imshow(sourceImage)
		cv2.imwrite('cave.png', sourceImage)
		cv2.destroyAllWindows()
		break
	(result, vis) = matched_result
	if split and i > 1:
		height = result.shape[0] if result.shape[0] > leftSourceImage.shape[0] else leftSourceImage.shape[0]
		width = result.shape[1] + leftSourceImage.shape[1]
		merge = np.zeros((height,width,3)).astype("uint8")
		merge[:leftSourceImage.shape[0], :leftSourceImage.shape[1]] = leftSourceImage
		merge[:result.shape[0], leftSourceImage.shape[1]:] = result
		result = merge
	# if i % 4 == 0:
	# 	result = ImageTransform.straighten(result, i)
	display_image = result
	if result.shape[1] > 1000:
		display_image = imutils.resize(display_image, width=1000)

	cv2.imshow('stitched', display_image)
	if match:
		cv2.imshow("Keypoint Matches", vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	sourceImage = result
	cv2.imwrite('cave.png', sourceImage)
