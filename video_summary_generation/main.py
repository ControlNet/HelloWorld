#!/usr/bin/env python
# coding: utf-8

# # Main Program

# In[1]:


get_ipython().run_cell_magic('time', '', '\nfrom collections import Counter\nfrom functools import partial\nfrom os import listdir\nfrom typing import List, Tuple, Optional, Type\n\nimport matplotlib.pyplot as plt\n# import ffmpeg\nimport numpy as np\nimport numpy.linalg as la\nimport tensorflow as tf\nimport tensorflow_hub as hub\nfrom numpy import ndarray\nfrom skimage import io\nfrom skimage.transform import rescale, resize\nfrom skimage.color import rgb2gray, rgb2hsv\nfrom skimage.feature import canny\n\n\ndef generateCoolSummary(images: List[ndarray]) -> Tuple[List[str], ndarray, ndarray]:\n    """\n    Args:\n        images (List[ndarray]): Extracted frames from a video.\n\n    Returns:\n        gifImages (List[ndarray]): The important frames used to build gif images.\n        collage (ndarray): The output collage.\n    """\n    # select frames\n    selectedFrameIndexes = FrameSelector(images)\n    selectedFrameIndexes.rescale((1, 0.05, 0.05, 1)).flatten().extractPrincipalComponents()\n    selectedFrameIndexes.clustering(k=4, Algorithm=KMeans)\n    selectedFrameIndexes = selectedFrameIndexes.getMostClosedIndex()\n    selectedFrames = np.array(images)[selectedFrameIndexes]\n\n    # get the overall score of importance\n    score = ImportanceDiscriminator(selectedFrames).computeOverallScore()\n    # sort\n    orderedImages = selectedFrames[np.argsort(score)[::-1]]\n    # generate collage and gif\n    collage = Collage(height=640, components=list(orderedImages), margin=50).merge()\n\n    gifFrames = createGif(orderedImages)\n    return gifFrames, collage.image\n\n\n# def extractFrames(videoPath, timeStrip=0., timePeriod=1., save=False):\n#     video = ffmpeg.input(videoPath)\n#     videoInfo = ffmpeg.probe(videoPath)\n#     videoLength = round(float(videoInfo["format"]["duration"]))\n#     videoFrameRate = round(eval(videoInfo["streams"][0]["avg_frame_rate"]))\n#     videoWidth = videoInfo["streams"][0]["width"]\n#     videoHeight = videoInfo["streams"][0]["height"]\n#     # strip the some second in the beginning and end\n#     # select framesInd with selected time period\n#     times = np.arange(timeStrip, videoLength - timeStrip, timePeriod)\n#     output, _ = video.output(\'pipe:\', format=\'rawvideo\', pix_fmt=\'rgb24\').run(capture_stdout=True)\n#     outputArray = np.frombuffer(output, np.uint8).reshape([-1, videoHeight, videoWidth, 3])\n#     framesInd = np.vectorize(round)(times * videoFrameRate).astype(int)\n#     frames = outputArray[framesInd]\n#     # resize\n#     frames = rescale(frames, scale=(1, 0.5, 0.5, 1))\n\n#     # save as files\n#     if save:\n#         for i, image in enumerate(frames):\n#             io.imsave(join("image", f"{i}.png"), arr=image)\n#     return frames\n\n\n# extractFrames("177_1.mp4", timePeriod=0.1, save=True)\n# read images from image directory\ndef loadImages() -> List[ndarray]:\n    return list(map(\n        lambda file: io.imread(f"image/{file}"),\n        sorted(listdir("image"), key=lambda x: int(x[:-4]))\n    ))\n\n\n# K-means implementation\nclass KMeans:\n    def __init__(self, k):\n        self.k = k\n        self.centroids: ndarray = np.array([])\n\n    def fit(self, x: ndarray, epochs=300):\n        # initialize the centroids\n        self.centroids = x[:self.k].copy()\n\n        prevCentroids = self.centroids.copy()\n        for epoch in range(epochs):\n            # E step\n            clusters = self.predict(x)\n\n            # M step\n            for eachK in range(self.k):\n                self.centroids[eachK] = x[clusters == eachK].mean(axis=0)\n\n            if (self.centroids == prevCentroids).all():\n                break\n            else:\n                prevCentroids = self.centroids.copy()\n\n        return self\n\n    def predict(self, x: ndarray) -> ndarray:\n        distance = self.calculateDistance(x)\n        clusters = distance.argmin(axis=1)\n        return clusters\n\n    def calculateDistance(self, x: ndarray):\n        distance = np.zeros((x.shape[0], self.k))\n        # calculate Euclidean distance for each cluster for each data point\n        for eachK in range(self.k):\n            distance[:, eachK] = ((x - self.centroids[eachK]) ** 2).sum(axis=1) ** 0.5\n        return distance\n\n\n# implementation of PCA\nclass PCA:\n    def __init__(self, dim: int = None):\n        self.dim = dim\n        self.mean_vec = 0\n        self.eigen_data = None\n        self.transform_matrix = None\n\n    def fit(self, x_train: ndarray):\n        self.mean_vec = np.mean(x_train, axis=0)\n        cov_matrix = np.cov((x_train - self.mean_vec).T)\n        eigenValue, eigenVec = la.eig(cov_matrix)\n        self.transform_matrix = eigenVec[:, :self.dim].astype(float)\n        return self\n\n    def predict(self, x_test: ndarray) -> ndarray:\n        return (x_test - self.mean_vec).dot(self.transform_matrix)\n\n\n# FrameSelector to select not similar frames\nclass FrameSelector:\n    def __init__(self, frames: List[ndarray]):\n        # down-sampling and reshaping the frames for faster computation\n        self.frames: ndarray = np.array(frames)\n        self.clusterer: Optional[KMeans] = None\n        self.pca = PCA(dim=2)\n\n    def resize(self, size):\n        self.frames = resize(self.frames, size).reshape(len(self.frames), -1)\n        return self\n\n    def rescale(self, scale):\n        self.frames = rescale(self.frames, scale).reshape(len(self.frames), -1)\n        return self\n\n    def flatten(self):\n        self.frames = self.frames.reshape(self.frames.shape[0], -1)\n        return self\n    \n    def extractPrincipalComponents(self):\n        self.pcaFrames = self.pca.fit(self.frames) \\\n            .predict(self.frames)\n        return self\n\n    def clustering(self, k: int, Algorithm: Type[KMeans]):\n        self.clusterer = Algorithm(k).fit(self.pcaFrames)\n        return self\n\n    def getMostClosedIndex(self):\n        distance = self.clusterer.calculateDistance(self.pcaFrames)\n        mostClosedIndex = np.argmin(distance, axis=0)\n        return mostClosedIndex\n\n\nclass ObjectDetector:\n    tags = np.array(\n        "person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop "\n        "sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "\n        "tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball glove skateboard surfboard tennis "\n        "racket bottle wine glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot dog pizza "\n        "donut cake chair couch potted plant bed dining table toilet tv laptop mouse remote keyboard cell phone "\n        "microwave oven toaster sink refrigerator book clock vase scissors teddy bear hair drier toothbrush" \\\n            .split(" ")\n    )\n\n    def __init__(self, url: str):\n        self.model = hub.load(url)\n        self.func = None\n\n    # select signature of TensorFlow-Hub model\n    def withSignature(self, signature: str):\n        self.func = self.model.signatures[signature]\n        return self\n\n    # predict\n    def __call__(self, input_):\n        return self.func(input_)\n\n\n# download and load the detector\ndetector = ObjectDetector("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1") \\\n    .withSignature("serving_default")\n\n\n# process one image to get number of persons\nclass ImageProcessor:\n\n    # only support single image\n    def __init__(self, image: ndarray):\n        self.image = image\n\n    # overall process for each frame image to get the info of boxes\' tags\n    def process(self, n, scoreThreshold, diagonalThreshold):\n        return self.withBoxesDetected() \\\n            .withBoxes(n, scoreThreshold, diagonalThreshold) \\\n            .getSummary()\n\n    def withImageToInt(self):\n        if self.image.dtype in [np.float32, np.float64]:\n            self.image = (self.image * 255).astype(np.uint8)\n        return self\n\n    def withBoxesDetected(self):\n        self.withImageToInt()\n        result = detector(tf.convert_to_tensor(self.image)[tf.newaxis, ...])\n        self.boxes = self.ImageBoxes(self, result)\n        return self\n\n    class ImageBoxes:\n        def __init__(self, obj, result):\n            self.image = obj.image\n            self.boxLocations = result["detection_boxes"][0].numpy()\n            self.tags = ObjectDetector.tags[result["detection_classes"][0].numpy().astype(int) - 1]\n            self.boxScores = result["detection_scores"][0].numpy()\n            self.nBox = self.boxScores.shape[0]\n\n        def getBoxInfo(self, i):\n            class BoxInfo:\n                def __init__(self, boxes):\n                    self.loc = boxes.boxLocations[i]\n                    self.tag = boxes.tags[i]\n                    self.score = boxes.boxScores[i]\n\n            return BoxInfo(self)\n\n        # input `n` as number of boxes to be cropped\n        def withBoxes(self, n, scoreThreshold, diagonalThreshold):\n            height, width, _ = self.image.shape\n\n            boxes = list(map(self.getBoxInfo, range(self.nBox)))\n            # sort by box score with descend order\n            boxes.sort(reverse=True, key=lambda box: box.score)\n            # type annotation alias\n            BoxInfo = boxes[0].__class__\n\n            # for each box, generate cropped image\n            def filterBox(box: BoxInfo) -> Optional[BoxInfo]:\n                # thresholding box score\n                if box.score < scoreThreshold:\n                    return None\n                # calculate the location of boxes for each edge\n                yMin, xMin, yMax, xMax = tuple(box.loc)\n                left, right, top, bottom = map(int, [xMin * width, xMax * width, yMin * height, yMax * height])\n                # thresholding the box size with diagonal length\n                if np.hypot(right - left, bottom - top) < diagonalThreshold:\n                    return None\n                return box\n\n            self.boxes: List[BoxInfo] = list(filter(\n                lambda box: box is not None,\n                map(filterBox, boxes[:min(n, self.nBox)])\n            ))\n            return self\n\n        def getBoxes(self):\n            return self.boxes\n\n    def withBoxes(self, n, scoreThreshold, diagonalThreshold):\n        self.croppedImages = self.boxes.withBoxes(n, scoreThreshold, diagonalThreshold)\n        return self\n\n    def getSummary(self):\n        boxes = self.boxes.getBoxes()\n        return Counter(map(lambda box: box.tag, boxes))\n\n\n# resize the images with specified height and proportional to width\ndef resizeByHeight(image: ndarray, height: int):\n    imageHeight, imageWidth, imageChannel = image.shape\n    return resize(image, output_shape=(height, int(imageWidth / (imageHeight / height)), imageChannel))\n\n\nclass Collage:\n\n    def __init__(self, height, components: List[ndarray], margin: int):\n        self.height = height + 2 * margin\n        self.margin = margin\n        # allocate components\n        confs, self.width = self.generateComponentConfs(components, margin)\n        # init the total collage\n        self.image = np.zeros((self.height, self.width, 3), dtype=np.float32)\n\n        # create each component\n        makeCollageComponentPartial = partial(makeCollageComponent, collage=self)\n        self.components = list(map(makeCollageComponentPartial, confs))\n\n    class CollageComponentConf:\n        def __init__(self, image: ndarray, leftTopLoc: Tuple[int, int]):\n            self.image = image\n            self.leftTopLoc = leftTopLoc\n\n        def __iter__(self):\n            return iter((self.image, self.leftTopLoc))\n\n    def generateComponentConfs(self, components: List[ndarray], margin: int) -> Tuple[List[CollageComponentConf], int]:\n        # determine if the component is in first row\n        firstRow = True\n        firstRowX = 0\n        secondRowX = 0\n        confs = []\n        for i, component in enumerate(components):\n            # resize the component\n            component = resizeByHeight(component, height=(self.height + margin) // 2)\n            # choose which row of the collage to place the component\n            leftTopY = 0 if firstRow else (self.height - margin) // 2\n            xPos = firstRowX if firstRow else secondRowX\n\n            # if it is the last one\n            if i == len(components) - 1:\n                # align the right edge of the total collage\n                leftTopX = (secondRowX if firstRow else firstRowX) - component.shape[1]\n            else:\n                leftTopX = max(0, xPos - margin)\n            confs.append(Collage.CollageComponentConf(component, (leftTopY, leftTopX)))\n            if firstRow:\n                if firstRowX == 0:\n                    firstRowX += margin\n                firstRowX += component.shape[1] - margin\n            else:\n                if secondRowX == 0:\n                    secondRowX += margin\n                secondRowX += component.shape[1] - margin\n            firstRow = firstRowX <= secondRowX\n\n        width = min(firstRowX, secondRowX)\n        return confs, width\n\n    @property\n    def shape(self):\n        return self.image.shape\n\n    def merge(self):\n        weights = np.array([*map(lambda component: component.weightMap, self.components)])\n        images = np.array([*map(lambda component: component.imageInCollage, self.components)])\n\n        for y in range(self.height):\n            for x in range(self.width):\n                if (weights[:, y, x] == 0).all():\n                    self.image[y, x] = 0\n                else:\n                    self.image[y, x] = np.average(\n                        a=images[:, y, x],\n                        weights=weights[:, y, x],\n                        axis=0\n                    )\n        return self\n\n    def getImage(self):\n        return self.image[self.margin:-self.margin, self.margin:-self.margin]\n\n\ndef makeCollageComponent(conf: Collage.CollageComponentConf, collage):\n    image, leftTopLoc = conf\n    return CollageComponent(image, leftTopLoc, collage)\n\n\nclass CollageComponent:\n    def __init__(self, image: ndarray, leftTopLoc: Tuple[int, int], collage: Collage):\n        # image numpy array\n        self.image = image\n        # calculate this component\'s height and width\n        self.imageHeight, self.imageWidth, _ = self.image.shape\n        # get left top corner coordinate, then calculate the coordinates of other 3 corners\n        self.leftTopLoc = np.array(leftTopLoc)\n        self.rightTopLoc = np.array((leftTopLoc[0], leftTopLoc[1] + self.imageWidth))\n        self.leftBottomLoc = np.array((leftTopLoc[0] + self.imageHeight, leftTopLoc[1]))\n        self.rightBottomLoc = np.array((leftTopLoc[0] + self.imageHeight, leftTopLoc[1] + self.imageWidth))\n        # calculate the center point\n        self.centerLoc = np.average(\n            [self.leftTopLoc, self.rightTopLoc, self.leftBottomLoc, self.rightBottomLoc], axis=0)\n        # set the collage which the component belongs to the\n        self.collage = collage\n        self.margin = collage.margin\n        # get occupyMap and weightMap\n        self.occupyMap = self.getOccupyMap()\n        self.weightMap = self.getWeightMap()\n        self.imageInCollage = self.getImageInCollage()\n\n    def getOccupyMap(self):\n        occupyMap = np.zeros((self.collage.height, self.collage.width))\n        occupyMap[self.leftTopLoc[0]:self.rightBottomLoc[0], self.leftTopLoc[1]:self.rightBottomLoc[1]] = 1\n        return occupyMap\n\n    def getWeightMap(self):\n        # generate gaussian square kernel\n        indexes = np.array(list([\n            [(y, x) for x in range(self.collage.shape[1])] for y in range(self.collage.shape[0])\n        ]))\n\n        def getWeightForLocation(loc):\n            if self.occupyMap[loc[0], loc[1]] == 0:\n                return 0.\n            y, x = loc\n            # distance to left and right edge\n            dLeftEdge = np.abs(x - self.leftTopLoc[1])\n            dRightEdge = np.abs(x - self.rightTopLoc[1])\n            dTopEdge = np.abs(y - self.leftTopLoc[0])\n            dBottomEdge = np.abs(y - self.leftBottomLoc[0])\n            # calculate the weight\n            leftWeight = min(1., dLeftEdge / self.margin)\n            rightWeight = min(1., dRightEdge / self.margin)\n            topWeight = min(1., dTopEdge / self.margin)\n            bottomWeight = min(1., dBottomEdge / self.margin)\n            # get the overall weight\n            return leftWeight * rightWeight * topWeight * bottomWeight\n\n        weightMap = np.apply_along_axis(\n            func1d=getWeightForLocation, arr=indexes, axis=2\n        )\n\n        return weightMap\n\n    def getImageInCollage(self):\n        imageInCollage = np.zeros((self.collage.height, self.collage.width, 3))\n        imageInCollage[self.leftTopLoc[0]:self.rightBottomLoc[0], self.leftTopLoc[1]:self.rightBottomLoc[1]] \\\n            = self.image\n        return imageInCollage\n\n\n# function to create gif frames\ndef createGif(images: ndarray, framesBetween: int = 10) -> ndarray:\n    previous = images[0]\n    finalFramesList = []\n    # for each image, calculate the linear interplation between this image and previous one\n    for image in images[1:]:\n        finalFramesList.append(np.linspace(previous, image, num=framesBetween, axis=0, endpoint=False))\n        previous = image.copy()\n\n    finalFramesList.append(images[-1][np.newaxis, ...])\n    return np.vstack(finalFramesList).astype(int)\n\n\n# The distriminator to calculate the importance score\nclass ImportanceDiscriminator:\n    def __init__(self, images: ndarray):\n        self.images = images\n\n    def computeEdgeComplexity(self, height=360, width=640):\n        # this function is to compute edge complexity by counting the number of edge pixels\n        grayImages = (rgb2gray(self.images) * 255).astype(np.uint8)\n        cannyArr = np.apply_along_axis(axis=1, arr=grayImages.reshape(self.images.shape[0], -1),\n                                       func1d=(lambda x: canny(x.reshape(height, width))))\n        # calculate the number of edge pixels\n        edgeComplexity = cannyArr.sum(axis=(1, 2))\n        # normalized\n        return edgeComplexity / edgeComplexity.max()\n\n    def computeLuminanceComplexity(self):\n        # calculate luminance variance to get luminance complexity score\n        grayImages = (rgb2gray(self.images) * 255).astype(np.uint8)\n        luminanceVar = np.var(grayImages.reshape(self.images.shape[0], -1), axis=1)\n        return luminanceVar / luminanceVar.max()\n\n    def computeColorComplexity(self):\n        # calculate the variance of hue values to get color complexity\n        # skimage in colab does not support batch of images\n        # so run it with mapping\n        hsvImages = np.stack(map(\n            lambda image: (rgb2hsv(image) * 255),\n            self.images\n        ), axis=3).astype(int)\n        # get hue, saturation and value separately\n        hImages = hsvImages[:, :, :, 0]\n        sImages = hsvImages[:, :, :, 1]\n        vImages = hsvImages[:, :, :, 2]\n        # calculate the variance of the multiplication of h, s and v\n        # when s is lower, the color becomes whiter; if v is lower, the color becomes darker\n        # so only when the saturation and value is high, the complexity of the colors are high\n        # so using the multiplication of hsv is the better choice to calculate the color complexity\n        # than hue-only or hsv-summation\n        hsvVar = np.var((hImages * sImages * vImages).reshape(self.images.shape[0], -1), axis=1)\n        return hsvVar / hsvVar.max()\n\n    def computePersonComplexity(self):\n        # calculate the person complexity score by the number of persons\n        # get the number of persons\n        numberOfPersons = np.array(list(map(lambda counter: counter.get("person", 0),\n                                            map(lambda each: each.process(n=100, scoreThreshold=0.95, diagonalThreshold=80),\n                                                map(ImageProcessor, self.images)))))\n        # map the score to range [0, 1]\n        adjusted = numberOfPersons - numberOfPersons.min()\n        return adjusted / adjusted.max()\n\n    def computeOverallScore(self):\n        edgeComplexity = self.computeEdgeComplexity(self.images.shape[1], self.images.shape[2])\n        luminanceComplexity = self.computeLuminanceComplexity()\n        colorComplexity = self.computeColorComplexity()\n        personComplexity = self.computePersonComplexity()\n        return edgeComplexity / 3 + luminanceComplexity / 3 + colorComplexity / 3 - personComplexity / 2\n\n\nimages = loadImages()\ngifFrames, collageImage = generateCoolSummary(images)')


# # Output

# In[2]:


plt.figure(figsize=(16, 5))
plt.imshow(collageImage)
plt.axis("off")
plt.show()


# In[3]:


plt.figure(figsize=(30, 8))
plt.suptitle("GIF Frames", y=0.9)
for i, gifFrame in enumerate(gifFrames):
    plt.subplot(4, 10, i + 1)
    plt.imshow(gifFrame)
    plt.axis("off")
plt.show()


# # Other plots for report
# 
# Only produce plots for report

# In[4]:


plt.figure(figsize=(24, 6))
plt.suptitle("Frames of Video")
for i, image in enumerate(images):
    plt.subplot(5, 10, i + 1)
    plt.imshow(image)
    plt.axis("off")
plt.show()


# In[5]:


# select frames
try:
    selector = FrameSelector(images)
    selector.rescale((1, 0.05, 0.05, 1)).flatten().extractPrincipalComponents()
except np.linalg.LinAlgError:
    # try again
    selector = FrameSelector(images)
    selector.rescale((1, 0.05, 0.05, 1)).flatten().extractPrincipalComponents()

selector.clustering(k=4, Algorithm=KMeans)

selectedFrameIndexes = selector.getMostClosedIndex()
selectedFrames = np.array(images)[selectedFrameIndexes]

# get the overall score of importance
score = ImportanceDiscriminator(selectedFrames).computeOverallScore()
# sort
orderedImages = selectedFrames[np.argsort(score)[::-1]]

collage = Collage(height=640, components=list(orderedImages), margin=50).merge()

gifFrames = createGif(orderedImages)


# In[6]:


plt.figure(figsize=(30, 15))
plt.suptitle("Frames of Video", y=0.95)
for i, image in enumerate(images):
    iRow = i // 10
    iCol = i % 10
    plt.subplot(10, 10, iRow * 20 + iCol + 1)
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(10, 10, iRow * 20 + iCol + 11)
    comp = selector.pcaFrames[i]
    plt.scatter(selector.pcaFrames[:, 0], selector.pcaFrames[:, 1], c="blue")
    plt.scatter(selector.pcaFrames[i, 0], selector.pcaFrames[i, 1], c="red")


# In[7]:


groups = selector.clusterer.predict(selector.pcaFrames)

plt.figure(figsize=(8, 5))
ax = plt.subplot(1, 1, 1)
plt.scatter(selector.pcaFrames[:, 0], selector.pcaFrames[:, 1], c=groups)
plt.scatter(selector.clusterer.centroids[:, 0], selector.clusterer.centroids[:, 1], marker="x", c=range(4))
plt.scatter(selector.pcaFrames[selectedFrameIndexes, 0], selector.pcaFrames[selectedFrameIndexes, 1], c="red")
plt.legend(["Frames", "Centroids", "Selected"])
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.title("The Selection of Frames")
ax.set_facecolor("#ACC3E8")


# In[8]:


plt.figure(figsize=(24, 10))
plt.suptitle("Selected Frames", y=0.65)
for i, each in enumerate(selectedFrames):
    plt.subplot(1, 4, i + 1)
    plt.imshow(each)
    plt.axis("off")
plt.show()


# In[9]:


personComplexity = ImportanceDiscriminator(selectedFrames).computePersonComplexity()

plt.figure(figsize=(24, 10))
plt.suptitle("Person Comlexity", y=0.65)
for i, each in enumerate(selectedFrames):
    plt.subplot(1, 4, i + 1)
    plt.imshow(each)
    plt.axis("off")
    plt.title(personComplexity[i])
plt.show()


# In[10]:


overallScore = ImportanceDiscriminator(selectedFrames).computeOverallScore()

plt.figure(figsize=(24, 10))
plt.suptitle("Importance Score", y=0.65)
for i, each in enumerate(selectedFrames):
    plt.subplot(1, 4, i + 1)
    plt.imshow(each)
    plt.axis("off")
    plt.title(round(overallScore[i],2))
plt.show()


# In[11]:


occupyMaps = list(map(lambda component: component.occupyMap, collage.components))
weightMaps = list(map(lambda component: component.weightMap, collage.components))
imageInCollage = list(map(lambda component: component.imageInCollage, collage.components))

plt.figure(figsize=(24, 13))
# plt.suptitle("Importance Score", y=0.65)
for i, each in enumerate(selectedFrames):
    plt.subplot(4, 4, i + 1)
    plt.imshow(each)
    plt.axis("off")
    plt.subplot(4, 4, i + 5)
    plt.imshow(occupyMaps[i], cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, i + 9)
    plt.imshow(weightMaps[i], cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, i + 13)
    plt.imshow(imageInCollage[i])
    plt.axis("off")
plt.show()


# In[ ]:




