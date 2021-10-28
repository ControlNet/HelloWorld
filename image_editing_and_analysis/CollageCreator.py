from multiprocessing import Pool, cpu_count
from os.path import join

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from skimage import io
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny
from skimage.transform import rescale


def CollageCreate(AddressofFolder):
    # read images
    images = readImages(AddressofFolder)
    # compute scores
    score = computeOverallScore(images)
    # sorting the score in descending order
    orderedImages = images[np.argsort(score)[::-1]]
    # make collage
    collage = makeCollage(orderedImages)
    io.imsave(join(AddressofFolder, "collage.png"), arr=collage.image)
    plt.imshow(collage.image)
    plt.axis("off")
    plt.show()
    return collage.image


def extractFrames(videoPath, timeStrip=10, timePeriod=3, save=False):
    video = ffmpeg.input(videoPath)
    videoInfo = ffmpeg.probe(videoPath)
    videoLength = round(float(videoInfo["format"]["duration"]))
    videoFrameRate = round(eval(videoInfo["streams"][0]["avg_frame_rate"]))
    videoWidth = videoInfo["streams"][0]["width"]
    videoHeight = videoInfo["streams"][0]["height"]
    # strip the 10 second in the beginning and end
    # select framesInd with selected time period
    times = np.array(list(range(timeStrip, videoLength - timeStrip, timePeriod)))
    output, _ = video.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
    outputArray = np.frombuffer(output, np.uint8).reshape([-1, videoHeight, videoWidth, 3])
    framesInd = times * videoFrameRate
    frames = outputArray[framesInd]
    # save as files
    if save:
        for i, image in enumerate(frames):
            io.imsave(join("image", f"{i}.png"), arr=image)
    return frames


# videoArr = extractFrames("video.mp4", save=True)

def readImages(folder):
    files = ["2.png", "12.png", "18.png", "35.png", "47.png"]
    images = np.array(list(map(lambda file: io.imread(join(folder, file)), files)))
    return images


def computeEdgeComplexity(images, height=268, width=640):
    # this function is to compute edge complexity by counting the number of edge pixels
    grayImages = (rgb2gray(images) * 255).astype(np.uint8)
    cannyArr = np.apply_along_axis(axis=1, arr=grayImages.reshape(images.shape[0], -1),
                                   func1d=(lambda x: canny(x.reshape(height, width))))
    # calculate the number of edge pixels
    edgeComplexity = cannyArr.sum(axis=(1, 2))
    # normalized
    return edgeComplexity / edgeComplexity.max()


def computeLuminanceComplexity(images):
    # calculate luminance variance to get luminance complexity score
    grayImages = (rgb2gray(images) * 255).astype(np.uint8)
    luminanceVar = np.var(grayImages.reshape(images.shape[0], -1), axis=1)
    return luminanceVar / luminanceVar.max()


def computeColorComplexity(images):
    # calculate the variance of hue values to get color complexity
    hsvImages = (rgb2hsv(images) * 255).astype(np.uint8)
    # get hue, saturation and value separately
    hImages = hsvImages[:, :, :, 0]
    sImages = hsvImages[:, :, :, 1]
    vImages = hsvImages[:, :, :, 2]
    # calculate the variance of the multiplication of h, s and v
    # when s is lower, the color becomes whiter; if v is lower, the color becomes darker
    # so only when the saturation and value is high, the complexity of the colors are high
    # so using the multiplication of hsv is the better choice to calculate the color complexity
    # than hue-only or hsv-summation
    hsvVar = np.var((hImages * sImages * vImages).reshape(images.shape[0], -1), axis=1)
    return hsvVar / hsvVar.max()


def computeOverallScore(images):
    edgeComplexity = computeEdgeComplexity(images, images.shape[1], images.shape[2])
    luminanceComplexity = computeLuminanceComplexity(images)
    colorComplexity = computeColorComplexity(images)
    return edgeComplexity + luminanceComplexity + colorComplexity


def makeCollage(orderedImages):
    leftTop, rightTop, leftBottom, rightBottom, center = orderedImages
    collage = Collage(800, 960, leftTop, rightTop, leftBottom, rightBottom, center).collage()
    return collage


def makeCollageComponent(args):
    return CollageComponent(*args)


class Collage:
    def __init__(self, height, width, leftTop, rightTop, leftBottom, rightBottom, center):
        self.height = height
        self.width = width
        self.image = np.zeros((height, width, 3))
        # normalize the images values as float in [0., 1.]
        leftTop = leftTop / 255
        rightTop = rightTop / 255
        leftBottom = leftBottom / 255
        rightBottom = rightBottom / 255
        center = rescale(center, scale=(1.5, 1.5, 1))
        # build collage components with multi-threading
        with Pool(min(cpu_count() - 1, 5)) as p:
            self.leftTop, self.rightTop, self.leftBottom, self.rightBottom, self.center = p.map(makeCollageComponent, [
                [leftTop, (0, 0), self],
                [rightTop, (0, self.width - rightTop.shape[1]), self],
                [leftBottom, (self.height - leftBottom.shape[0], 0), self],
                [rightBottom, (self.height - rightBottom.shape[0], self.width - rightBottom.shape[1]), self],
                [center, (self.height // 2 - center.shape[0] // 2, 0), self]
            ])

    def collage(self):
        for y in range(self.height):
            for x in range(self.width):
                self.image[y, x] = np.average(
                    a=[self.leftTop.imageInCollage[y, x], self.rightTop.imageInCollage[y, x],
                       self.leftBottom.imageInCollage[y, x], self.rightBottom.imageInCollage[y, x],
                       self.center.imageInCollage[y, x]],
                    weights=[self.leftTop.weightMap[y, x], self.rightTop.weightMap[y, x],
                             self.leftBottom.weightMap[y, x],
                             self.rightBottom.weightMap[y, x], self.center.weightMap[y, x]],
                    axis=0
                )
        return self


class CollageComponent:
    def __init__(self, image, leftTopLoc, collage):
        # image numpy array
        self.image = image
        # calculate this component's height and width
        self.imageHeight, self.imageWidth, _ = self.image.shape
        # get left top corner coordinate, then calculate the coordinates of other 3 corners
        self.leftTopLoc = np.array(leftTopLoc)
        self.rightTopLoc = np.array((leftTopLoc[0], leftTopLoc[1] + self.imageWidth))
        self.leftBottomLoc = np.array((leftTopLoc[0] + self.imageHeight, leftTopLoc[1]))
        self.rightBottomLoc = np.array((leftTopLoc[0] + self.imageHeight, leftTopLoc[1] + self.imageWidth))
        # calculate the center point
        self.centerLoc = np.average(
            [self.leftTopLoc, self.rightTopLoc, self.leftBottomLoc, self.rightBottomLoc], axis=0)
        # set the collage which the component belongs to the
        self.collage: Collage = collage
        # get occupyMap and weightMap
        self.occupyMap = self.getOccupyMap()
        self.weightMap = self.getWeightMap()
        self.imageInCollage = self.getImageInCollage()

    def getOccupyMap(self):
        occupyMap = np.zeros((self.collage.height, self.collage.width))
        occupyMap[self.leftTopLoc[0]:self.rightBottomLoc[0],
                  self.leftTopLoc[1]:self.rightBottomLoc[1]] = 1
        return occupyMap

    def getWeightMap(self, sigma=3000):
        weightMap = self.occupyMap.copy()
        horizontalSigma = sigma / self.imageHeight * self.imageWidth
        gaussianDistribution = multivariate_normal(mean=self.centerLoc, cov=[[sigma, 0], [0, horizontalSigma]])
        # generate gaussian square kernel
        gaussianDistributionMapIndex = np.array([
            [(x, y) for y in range(weightMap.shape[1])] for x in range(weightMap.shape[0])
        ])
        gaussianDistributionMap = np.apply_along_axis(
            func1d=gaussianDistribution.pdf, arr=gaussianDistributionMapIndex, axis=2
        )
        # only the occupied pixels has non-zero weights
        weightMap = self.occupyMap * gaussianDistributionMap
        # normalize
        weightMap = weightMap / weightMap.max()
        return weightMap

    def getImageInCollage(self):
        imageInCollage = np.zeros((self.collage.height, self.collage.width, 3))
        imageInCollage[
        self.leftTopLoc[0]:self.rightBottomLoc[0], self.leftTopLoc[1]:self.rightBottomLoc[1]
        ] = self.image
        return imageInCollage


if __name__ == '__main__':
    collage = CollageCreate("image")
