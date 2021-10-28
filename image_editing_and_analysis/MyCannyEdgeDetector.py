from functools import partial
from multiprocessing import Pool, cpu_count
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.stats import multivariate_normal
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters.edges import convolve
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def myCannyEdgeDetector(image, Low_Threshold, High_Threshold):
    # read from path
    image = readImage(image)
    return image.gaussianFilter().computeGradient().nonMaximumSuppression() \
        .applyThreshold(lowThreshold=Low_Threshold, highThreshold=High_Threshold).values


class Image:
    def __init__(self, values: ndarray):
        self.values = values
        self.theta = np.zeros(values.shape)

    def show(self, cmap="gray"):
        plt.imshow(self.values, cmap=cmap)
        plt.show()

    # a function transform the image from (0, 255) to (0., 1.)
    def toInt(self):
        self.values = (self.values * 255).astype(int)
        return self

    # a function inverse the transform from (0., 1.) to (0, 255)
    def toFloat(self):
        self.values = self.values / self.values.max()
        return self

    # a function normalize the scale in (0, 255)
    def normalized(self):
        return self.toFloat().toInt()

    def toBool(self):
        self.values = self.values.astype(bool)
        return self

    # zero padding to keep convolution computation in same size
    def zeroPadding(self, pad: int):
        self.values = np.pad(self.values, pad_width=pad)
        return self

    # convolution computation
    def convolutionTo(self, kernel: ndarray):
        return convolve(self.values, weights=kernel)

    # gaussian filter to smooth image
    def gaussianFilter(self, kernelSize=3, sigma=1.):
        # set 2D gaussian distribution
        centerPos = kernelSize // 2
        gaussianDistribution = multivariate_normal(mean=[centerPos, centerPos], cov=[[sigma, 0], [0, sigma]])
        # generate gaussian square kernel
        gaussianKernelIndex = np.array([
            [(y, x) for x in range(kernelSize)] for y in range(kernelSize)
        ])
        gaussianKernel = np.apply_along_axis(func1d=gaussianDistribution.pdf, arr=gaussianKernelIndex, axis=2)
        # apply convolution
        self.values = self.convolutionTo(gaussianKernel)
        return self.normalized()

    def computeGradient(self):
        # define the sobel kernels
        xSobelKernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        ySobelKernel = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        # apply convolution for the x-axis and y-axis
        xGrad = self.convolutionTo(xSobelKernel).astype("float64")
        yGrad = self.convolutionTo(ySobelKernel).astype("float64")
        # calculate overall gradient
        grad = np.hypot(xGrad, yGrad)
        self.values = grad
        self.theta = np.arctan2(yGrad, xGrad)
        return self.normalized()

    def nonMaximumSuppression(self):
        # calculate orientationMap maps from the results of computeGradient
        @np.vectorize
        def normalize_angle(angle):
            return angle + 180 if angle < 0 else angle

        orientationMap = normalize_angle(self.theta * 180 / np.pi)
        # As the local maximum need to compare the nearby pixels, the most outside pixels are not contained
        pixelLocations = np.array([
            [(y, x) for x in range(1, self.values.shape[1] - 1)] for y in range(1, self.values.shape[0] - 1)
        ])

        # define an inner function to non-maximum-suppression for each magnitude
        def nonMaximumSuppressionForPixel(pixelLoc):
            # calculate magnitude and orientation
            y, x = pixelLoc
            magnitude = self.values[y, x]
            orientation = orientationMap[y, x]
            # get nearby pixels: near1, near2
            # in horizon direction
            if 0 <= orientation < 22.5 or 157.5 <= orientation <= 180:
                near1 = self.values[y, x + 1]
                near2 = self.values[y, x - 1]
            # in diagonal direction
            elif 22.5 <= orientation < 67.5:
                near1 = self.values[y + 1, x - 1]
                near2 = self.values[y - 1, x + 1]
            # in vertical direction
            elif 67.5 <= orientation < 112.5:
                near1 = self.values[y + 1, x]
                near2 = self.values[y - 1, x]
            # in another diagonal direction
            elif 112.5 <= orientation < 157.5:
                near1 = self.values[y - 1, x - 1]
                near2 = self.values[y + 1, x + 1]
            else:
                return 0
            # if the magnitude value is higher than nearby pixels, it is local maximum
            if magnitude >= near1 and magnitude >= near2:
                return magnitude
            else:
                return 0

        # map each location to the images to perform non-maximum-suppression
        self.values = np.apply_along_axis(arr=pixelLocations, axis=2, func1d=nonMaximumSuppressionForPixel)
        # the most outside pixels are gone, so need to be zero padded.
        return self.zeroPadding(1).toFloat()

    def applyThreshold(self, lowThreshold, highThreshold):
        strongMap = self.values >= highThreshold
        weakMap = (self.values < highThreshold) & (self.values >= lowThreshold)

        # define a function to check if the weak points have nearby strong points
        def checkNeighbours(weakLoc):
            # check if one of nearby locations is strong
            neighbours = getNeighbours(weakLoc, strongMap.shape)

            def checkStrong(loc):
                y, x = loc
                return strongMap[y, x]

            hasStrongNearby = np.apply_along_axis(func1d=checkStrong, axis=1, arr=neighbours).any()
            # if there is a strong edge nearby, mark the pixel as a strong edge.
            if hasStrongNearby:
                y, x = weakLoc
                strongMap[y, x] = True
            return None

        # a function to get all neighbours' locations
        def getNeighbours(loc, imageShape):
            y, x = loc
            max_y, max_x = imageShape
            neighboursLocations = [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1),
                                   (y + 1, x + 1), (y - 1, x - 1), (y + 1, x - 1), (y - 1, x + 1)]
            neighboursLocations = filter(
                lambda coordinate: coordinate[0] in range(max_y) and coordinate[1] in range(max_x),
                neighboursLocations)
            return np.array(list(neighboursLocations))

        # get all weak pixel locations for iteration
        weakLocations = np.vstack(np.where(weakMap)).T
        np.apply_along_axis(func1d=checkNeighbours, axis=1, arr=weakLocations)
        self.values = strongMap
        return self


# read image files output as an Image object
def readImage(path: str) -> Image:
    return Image(rgb2gray(imread(path)))


def calculatePsnrAndSsim(lowThres, highThres, skimageOutput):
    if highThres > lowThres:
        try:
            myOutput = myCannyEdgeDetector(join("image", "task1.png"),
                                           Low_Threshold=lowThres, High_Threshold=highThres)
        except ValueError:
            return 0, 0
        else:
            psnr = peak_signal_noise_ratio(image_true=skimageOutput, image_test=myOutput)
            ssim = structural_similarity(im1=skimageOutput, im2=myOutput)
            return psnr, ssim
    else:
        return 0, 0


@np.vectorize
def round2(n):
    return round(n, 2)


@np.vectorize
def round3(n):
    return round(n, 3)


def thresholdGridSearch():
    # apply functions to the task1.png
    skimageOutput = canny(readImage(join("image", "task1.png")).values)
    lowThresholds = round2(np.arange(0.01, 0.2, 0.01))
    highThresholds = round2(np.arange(0.02, 0.5, 0.02))
    psnr = []
    ssim = []
    p = Pool(cpu_count() // 2)
    for lowThres in lowThresholds:
        calculatePsnrAndSsimFunc = partial(calculatePsnrAndSsim, lowThres=lowThres, skimageOutput=skimageOutput)
        results = np.array(list(p.map(calculatePsnrAndSsimFunc, highThresholds)))
        psnr.append(results[:, 0])
        ssim.append(results[:, 1])

    psnr = np.array(psnr) / np.max(psnr)
    ssim = np.array(ssim)
    # plot two results
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.set_yticks(range(len(lowThresholds)))
    ax1.set_yticklabels(lowThresholds)
    ax1.set_xticks(range(len(highThresholds)))
    ax1.set_xticklabels(highThresholds)
    ax1.imshow(psnr, cmap="RdYlGn")

    ax2.set_yticks(range(len(lowThresholds)))
    ax2.set_yticklabels(lowThresholds)
    ax2.set_xticks(range(len(highThresholds)))
    ax2.set_xticklabels(highThresholds)
    ax2.imshow(ssim, cmap="RdYlGn")
    plt.show()


def lowThresholdSearch():
    skimageOutput = canny(readImage(join("image", "task1.png")).values)
    lowThresholds = round3(np.arange(0.01, 0.137, 0.002))
    p = Pool(cpu_count() // 2)
    calculatePsnrAndSsimFunc = partial(calculatePsnrAndSsim, highThres=0.14, skimageOutput=skimageOutput)
    results = np.array(list(p.map(calculatePsnrAndSsimFunc, lowThresholds)))
    psnr = np.array(results[:, 0])
    ssim = np.array(results[:, 1])
    # plot two results
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(lowThresholds, psnr, c="red")
    ax2.plot(lowThresholds, ssim, c="blue")
    plt.legend(["PSNR(Normalized)", "SSIM"])
    plt.show()


# main
def main():
    # apply functions to the task1.png
    myOutput = myCannyEdgeDetector(join("image", "task1.png"), Low_Threshold=0.093, High_Threshold=0.14)
    skimageOutput = canny(readImage(join("image", "task1.png")).values)

    # compute PSNR and SSIM
    psnr = peak_signal_noise_ratio(image_true=skimageOutput, image_test=myOutput)
    ssim = structural_similarity(im1=skimageOutput, im2=myOutput)

    # plot two results
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.imshow(skimageOutput, cmap="gray")
    ax1.set_title("output of skimage.feature.canny")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.imshow(myOutput, cmap="gray")
    ax2.set_title("output of myCannyEdgeDetector")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    plt.suptitle(f"PSNR: {psnr}\nSSIM: {ssim}", y=0.1)
    print(f"PSNR: {psnr}\nSSIM: {ssim}")
    plt.show()


if __name__ == '__main__':
    main()
