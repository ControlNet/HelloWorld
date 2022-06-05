from __future__ import annotations

import os.path

from sklearnex import patch_sklearn

patch_sklearn(verbose=False)

from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional
from pathlib import Path
import time

# import libraries
import cv2  # SIFT
import numpy as np
from numpy import ndarray
from scipy.stats import mode
from tensorflow.keras.datasets import cifar10
from tqdm.auto import trange
from enum import Enum
import time
from skimage.transform import rescale
from scipy.spatial.distance import cdist


class DistanceType(Enum):
    """
    Enum for distance type
    """
    L1 = "L1"
    L2 = "L2"

    @classmethod
    def of(cls, type_str: str) -> DistanceType:
        if type_str == cls.L1.value:
            return cls.L1
        elif type_str == cls.L2.value:
            return cls.L2
        else:
            raise ValueError("Invalid distance type")


class KMeans:

    def __init__(self, k: int, distance_type: DistanceType = DistanceType.L2, max_workers: int = 0):
        self.k = k
        self.centroids: ndarray = np.array([])
        self.distance_type = distance_type
        self.max_workers = max_workers
        if max_workers > 0:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = None

    def fit(self, x: ndarray, epochs=300):
        # initialize the centroids
        self.centroids = x[np.random.choice(len(x), size=self.k)].copy()

        prev_centroids = self.centroids.copy()
        for epoch in trange(epochs):
            # E step
            clusters = self.predict(x)

            # M step
            for each_k in range(self.k):
                self.centroids[each_k] = x[clusters == each_k].mean(axis=0)

            if np.abs(self.centroids - prev_centroids).sum() < 1:
                break
            else:
                prev_centroids = self.centroids.copy()

        return self

    def predict(self, x: ndarray, force_non_parallel: bool = False) -> ndarray:
        if self.max_workers > 0 and not force_non_parallel:
            futures = []
            partition_length = int(len(x) / self.max_workers)
            for i in range(self.max_workers):
                if i == self.max_workers - 1:
                    futures.append(self.executor.submit(
                        self._calculate_distance, x[i * partition_length:], self.centroids, self.distance_type)
                    )
                else:
                    futures.append(self.executor.submit(
                        self._calculate_distance, x[i * partition_length:(i + 1) * partition_length], self.centroids,
                        self.distance_type)
                    )

            distance = []
            for future in futures:
                distance.append(future.result())

            distance = np.concatenate(distance, axis=0)
        else:
            distance = self.calculate_distance(x)
        clusters = distance.argmin(axis=1)
        return clusters

    @staticmethod
    def _calculate_distance_l2(x: ndarray, centroids: ndarray) -> ndarray:
        return cdist(x, centroids, 'euclidean')

    @staticmethod
    def _calculate_distance_l1(x: ndarray, centroids: ndarray) -> ndarray:
        return cdist(x, centroids, 'cityblock')

    @classmethod
    def _calculate_distance(cls, x: ndarray, centroids: ndarray, distance_type: DistanceType) -> ndarray:
        if distance_type == DistanceType.L1:
            return cls._calculate_distance_l1(x, centroids)
        elif distance_type == DistanceType.L2:
            return cls._calculate_distance_l2(x, centroids)

    def calculate_distance(self, x: ndarray) -> ndarray:
        return self._calculate_distance(x, self.centroids, self.distance_type)

    def save(self, path: str = ".") -> None:
        # mkdir
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(path, "dictionary.npy"), self.centroids)

    @classmethod
    def load(cls, path: str, distance_type: DistanceType = DistanceType.L2, max_workers: int = 0) -> KMeans:
        dictionary = np.load(os.path.join(path, "dictionary.npy"))
        obj = cls(k=dictionary.shape[0], distance_type=distance_type, max_workers=max_workers)
        obj.centroids = dictionary
        return obj


def detect_sift(data: ndarray, contrast_threshold: float = 0.04, sigma: float = 1.6, scale: float = 2.0
) -> List[ndarray]:
    sift = cv2.SIFT_create(contrastThreshold=contrast_threshold, sigma=sigma)
    descriptors = []
    for i in trange(len(data)):
        img = (rescale(data[i], (scale, scale, 1)) * 255).astype(np.uint8)
        desc = sift.detectAndCompute(img, None)[1]
        if desc is not None:
            descriptors.append(desc)
        else:
            descriptors.append(np.zeros((0, 128)))

    return descriptors


def create_dictionary(k: int, distance_type: DistanceType = DistanceType.L2, max_workers: int = 0, epochs: int = 100
) -> KMeans:
    model = KMeans(k, distance_type=distance_type, max_workers=max_workers)
    model.fit(train_descriptors_all, epochs=epochs)
    return model


def create_histogram(img_descriptors: ndarray, k_means: KMeans) -> ndarray:
    if img_descriptors.shape[0] > 0:
        clusters = k_means.predict(img_descriptors, force_non_parallel=True)
        # clusters = k_means.predict(img_descriptors)
        return np.bincount(clusters, minlength=k_means.k) / img_descriptors.shape[0]
    else:
        return np.zeros(k_means.k)


def match_histogram(histogram_1: ndarray, target_histograms: ndarray, distance_type: DistanceType = DistanceType.L1
) -> ndarray:
    # L1 distance
    if distance_type == DistanceType.L1:
        return np.abs(histogram_1 - target_histograms).sum(axis=1)
    elif distance_type == DistanceType.L2:
        return ((histogram_1 - target_histograms) ** 2).sum(axis=1)


def predict_knn(x_test: ndarray, x_train: ndarray, y_train, k: int, distance_type: DistanceType = DistanceType.L1
) -> int:
    # for single x_test
    distances = match_histogram(x_test, x_train, distance_type)
    return mode(y_train[np.argsort(distances)[:k]])[0].squeeze()


def predict_knn_for_partition(x_test: ndarray, x_train: ndarray, y_train: ndarray, k: int,
    distance_type: DistanceType = DistanceType.L1, i = 0
) -> ndarray:
    # for multiple x_test
    pred_test = np.zeros(x_test.shape[0])
    for i in trange(x_test.shape[0], position=i):
        pred_test[i] = predict_knn(x_test[i], x_train, y_train, k, distance_type)
    return pred_test


def predict_knn_for_dataset(x_test: ndarray, x_train: ndarray, y_train, k: int, max_workers: int,
    distance_type: DistanceType = DistanceType.L1
) -> ndarray:
    # for all test dataset
    if max_workers > 0:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            partition_length = int(len(x_test) / max_workers)
            for i in range(max_workers):
                if i == max_workers - 1:
                    futures.append(executor.submit(
                        predict_knn_for_partition, x_test[i * partition_length:],
                        x_train, y_train, k, distance_type, i
                    ))
                else:
                    futures.append(executor.submit(
                        predict_knn_for_partition, x_test[i * partition_length:(i + 1) * partition_length],
                        x_train, y_train, k, distance_type, i
                    ))

            pred_test_list = []
            for future in futures:
                pred_test_list.append(future.result())
            pred_test = np.concatenate(pred_test_list)

    else:
        pred_test = np.zeros(len(x_test))
        for i in trange(len(x_test)):
            pred_test[i] = predict_knn(x_test[i], x_train, y_train, k, distance_type)

    return pred_test


def evaluate(y_pred: ndarray, y_test: ndarray) -> float:
    return (y_pred == y_test.squeeze()).mean()


if __name__ == '__main__':
    t0 = time.time()

    hyperparameter_defaults = {
        "img_scale": 2.1,
        "sift_contrast_threshold": 0.017,
        "sift_sigma": 1.98,
        "k_means_k": 674,
        "k_means_distance": "L2",
        "knn_k": 205,
        "knn_distance": "L2"
    }
    config = hyperparameter_defaults

    np.random.seed(42)

    # parameters
    MODEL_PATH: Optional[str] = ""
    NUM_WORKERS = 10
    IMG_SCALE = config["img_scale"]
    SIFT_CONTRAST_THRESHOLD = config["sift_contrast_threshold"]
    SIFT_SIGMA = config["sift_sigma"]
    K_MEANS_K = config["k_means_k"]
    K_MEANS_EPOCH = 300
    K_MEANS_DISTANCE = config["k_means_distance"]
    KNN_K = config["knn_k"]
    KNN_DISTANCE = config["knn_distance"]

    # Load the dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # sift detection
    train_descriptors = detect_sift(train_images, contrast_threshold=SIFT_CONTRAST_THRESHOLD, sigma=SIFT_SIGMA,
                                    scale=IMG_SCALE)
    test_descriptors = detect_sift(test_images, contrast_threshold=SIFT_CONTRAST_THRESHOLD, sigma=SIFT_SIGMA,
                                   scale=IMG_SCALE)
    train_descriptors_all = np.concatenate(train_descriptors, axis=0)

    # k-means clustering
    if MODEL_PATH is None:
        k_means = create_dictionary(K_MEANS_K, DistanceType.of(K_MEANS_DISTANCE), NUM_WORKERS, K_MEANS_EPOCH)
        k_means.save(".")
    else:
        k_means = KMeans.load(MODEL_PATH)

    # compute histograms as features
    train_features = np.zeros((len(train_images), k_means.k))
    test_features = np.zeros((len(test_images), k_means.k))

    for i in range(len(train_features)):
        train_features[i] = create_histogram(train_descriptors[i], k_means)

    for i in range(len(test_features)):
        test_features[i] = create_histogram(test_descriptors[i], k_means)

    # predict labels by KNN
    test_preds = predict_knn_for_dataset(test_features, train_features, train_labels, KNN_K, NUM_WORKERS)

    # evaluate
    print("Accuracy", evaluate(test_preds, test_labels))
    print("Runtime", time.time() - t0)
