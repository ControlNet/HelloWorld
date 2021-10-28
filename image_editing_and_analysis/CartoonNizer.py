import os
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import rescale


def CartoonNizer(image):
    original = CartoonNizer(image)
    cartoonized = original.cartoonize()

    plot_two_images(cartoonized, original)
    return cartoonized.value


# a function applying bilateral filter for each row for multi-threading programming
def _apply_to_row(row_index, self, kernel_size, sigma_d, sigma_r):
    height, width, channel = self.value.shape
    padding = kernel_size // 2
    row_filtered = []
    for x in range(padding, width - padding):
        row_filtered.append(
            self.apply_bilateral_filter(
                self.value[row_index - padding:row_index + padding + 1, x - padding:x + padding + 1],
                sigma_d=sigma_d, sigma_r=sigma_r
            )
        )
    return np.array(row_filtered)


class CartoonNizer:
    def __init__(self, image):
        self.value = image

    def copy(self):
        return CartoonNizer(self.value)

    def cartoonize(self):
        # get image after bilateral filter
        filtered = self \
            .rescale((1 / 3, 1 / 3, 1.), anti_aliasing=True) \
            .bilateral_filter(kernel_size=11, sigma_d=100, sigma_r=0.01)
        # detecting edges to the filtered images
        edges = filtered \
            .rescale((3, 3, 1)) \
            .detect_edges()
        # apply k-means clustering and add edges to the filtered images
        result = filtered.color_clustering(32, max_epoch=30) \
            .rescale((3, 3, 1)) \
            .add_edges(edges)
        return result

    def rescale(self, scale, *args, **kwargs):
        value = rescale(self.value, scale=scale, *args, **kwargs)
        return CartoonNizer(value)

    def show(self):
        plt.imshow(self.value)
        plt.show()

    def bilateral_filter(self, kernel_size, sigma_d, sigma_r):
        print("Bilateral filtering")
        height, width, channel = self.value.shape

        padding = kernel_size // 2
        value = np.pad(self.value, pad_width=padding)[:, :, padding: - padding]

        # convolution
        apply_to_row_func = partial(_apply_to_row, self=CartoonNizer(value), kernel_size=kernel_size,
                                    sigma_d=sigma_d, sigma_r=sigma_r)
        with Pool(cpu_count() - 2) as p:
            filtered = np.array(list(p.map(apply_to_row_func, range(padding, height + padding))))

        value = np.array(filtered)
        return CartoonNizer(value)

    @classmethod
    def apply_bilateral_filter(cls, window, sigma_d, sigma_r):
        # window is a windowed part of original image
        # get each channel
        r = window[:, :, 0]
        g = window[:, :, 1]
        b = window[:, :, 2]

        indexes = np.array([
            [(y, x) for x in range(window.shape[1])] for y in range(window.shape[0])
        ])

        # calculate weight for each weight
        def weight_of(window_channel):
            def wrapper(args):
                i, j = args
                k = window_channel.shape[0] // 2
                l = window_channel.shape[1] // 2
                f = lambda i, j: window_channel[i, j]
                spatial_weight = np.exp(-((i - k) ** 2 + (j - l) ** 2) / (2 * (sigma_d ** 2)))
                range_weight = np.exp(-(f(i, j) - f(k, l)) ** 2 / (2 * (sigma_r ** 2)))
                return spatial_weight * range_weight

            return wrapper

        weight_r = np.apply_along_axis(axis=2, arr=indexes, func1d=weight_of(r))
        weight_g = np.apply_along_axis(axis=2, arr=indexes, func1d=weight_of(g))
        weight_b = np.apply_along_axis(axis=2, arr=indexes, func1d=weight_of(b))
        # normalize
        weight_r = weight_r / weight_r.sum()
        weight_g = weight_g / weight_g.sum()
        weight_b = weight_b / weight_b.sum()
        # convolution for each channel
        return np.array([
            np.sum(r * weight_r),
            np.sum(g * weight_g),
            np.sum(b * weight_b)
        ])

    def color_clustering(self, k: int, max_epoch: int = 1000):
        print("K-means Clustering")
        height, width, channel = self.value.shape
        # flatten all pixels as RGB 3-dimensional vectors as training dataset
        pixels = self.value.reshape(-1, channel)
        # train the K-Means model
        k_means = KMeans(k).fit(pixels, max_epoch)
        # predict clusters to the input image
        clusters = k_means.predict(pixels)
        centroids = k_means.centroid

        # assign the centroids to each pixel belonging to each cluster
        clustered_image = np.apply_along_axis(func1d=lambda x: centroids[x], axis=0, arr=clusters)
        value = clustered_image.reshape(height, width, channel)
        return CartoonNizer(value)

    def detect_edges(self, *args, **kwargs):
        height, width, channel = self.value.shape
        edge = canny(rgb2gray(self.value), *args, **kwargs).astype(int)
        return edge.reshape(height, width, 1)

    def add_edges(self, edge):
        value = self.value - edge

        # set negative value as 0.
        @np.vectorize
        def normalize_negative_values(pixel):
            return 0. if pixel < 0 else pixel
        return CartoonNizer(normalize_negative_values(value))


# k means implementation
class KMeans:
    def __init__(self, k: int):
        self.k = k
        self.centroid = np.array([])

    def predict_cluster(self, row):
        distance = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=row - self.centroid)
        return np.argmin(distance)

    def fit(self, x, max_epochs: int = 1000):
        # m: number of rows, d: number of dimensions
        m, d = x.shape
        row_indexes = np.arange(m)
        # random choose k centroids
        np.random.shuffle(row_indexes)
        self.centroid = x[row_indexes[:self.k]]
        # repeat max_epochs times
        for epoch in range(max_epochs):
            print(f"K-means epoch: {epoch + 1} / {max_epochs}")
            centroid_old = self.centroid.copy()
            # E step
            # calculate each row to those centroids
            cluster = self.predict(x)
            # M step
            for each_cluster in range(self.k):
                self.centroid[each_cluster] = x[cluster == each_cluster].mean(axis=0)

            if np.array_equal(centroid_old, self.centroid):
                break

        return self

    def predict(self, x):
        with Pool(cpu_count() - 2) as p:
            res = np.array(list(p.map(self.predict_cluster, x)))
        return res
        # return np.apply_along_axis(func1d=self.predict_cluster, axis=1, arr=x)


def plot_two_images(cartoonized: CartoonNizer, original: CartoonNizer):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.imshow(original.value)
    ax1.set_title("Original")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.imshow(cartoonized.value)
    ax2.set_title("Cartoonized")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    plt.show()


if __name__ == '__main__':
    # res = CartoonNizer(plt.imread(os.path.join("image", "woman-4525714_1280.jpg")))
    res = CartoonNizer(plt.imread(os.path.join("image", "image-asset.jpeg")))
