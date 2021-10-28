from functools import partial
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.filters.edges import convolve

from multiprocessing import Pool


def CountShapes(image):
    # build ShapeCounter object
    image = ShapeCounter(image)
    with Pool(2) as p:
        apply_detector_f = partial(apply_detector, image=image)
        circles, spheres = p.map(apply_detector_f, iterable=[detect_circles, detect_spheres])

    circles_number = circles.value.shape[0]
    spheres_number = spheres.value.shape[0]
    total_number = circles_number + spheres_number

    output = image.add_circle_layers(circles, circle_thickness=1, center_thickness=5) \
        .add_circle_layers(spheres, circle_thickness=1, center_thickness=0) \
        .value
    plt.imshow(output)
    plt.title(f"Circles: {circles_number}, Spheres: {spheres_number}, Total: {total_number}")
    plt.axis("off")
    plt.savefig(os.path.join("image", "CountShapesOutput.png"))
    plt.show()
    return total_number


# for multithreading
def apply_detector(f, image):
    return f(image)


def detect_circles(image):
    circles = image.detect_edge(low_threshold=0.1276, high_threshold=1.03) \
        .find_potential_centers(center_score_threshold=13, min_radius=16, max_radius=80) \
        .sort_centers() \
        .find_circles(center_distance_threshold=58, radius_score_threshold=9.5,
                      radius_distance_threshold=4)
    return circles


def detect_spheres(image):
    spheres = image.detect_edge(low_threshold=0.5479, high_threshold=0.9215) \
        .find_potential_centers(center_score_threshold=2, min_radius=8, max_radius=16) \
        .sort_centers() \
        .find_circles(center_distance_threshold=33, radius_score_threshold=4.848,
                      radius_distance_threshold=10, detect_concentric_circles=False)
    return spheres


class ShapeCounter:
    def __init__(self, image, x_grad=None, y_grad=None, centers_loc=None,
                 max_radius=None, min_radius=None, edge_points=None):
        self.value = image
        self.x_grad = x_grad
        self.y_grad = y_grad
        self.centers_loc = centers_loc
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.edge_points = edge_points

    def show(self, cmap=None):
        if cmap is None:
            plt.imshow(self.value)
        else:
            plt.imshow(self.value, cmap=cmap)
        plt.show()

    def detect_edge(self, low_threshold=0.3, high_threshold=0.6):
        gray_image = rgb2gray(self.value)
        edge = canny(gray_image, sigma=1.5, low_threshold=low_threshold, high_threshold=high_threshold)
        # calculate the gradient in x and y direction
        x_sobel_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        y_sobel_kernel = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        # apply convolution for the x-axis and y-axis
        smoothed = gaussian(gray_image, sigma=1.5)
        x_grad = convolve(smoothed, weights=x_sobel_kernel).astype("float64")
        y_grad = convolve(smoothed, weights=y_sobel_kernel).astype("float64")
        return ShapeCounter(edge, x_grad=x_grad, y_grad=y_grad)

    def find_potential_centers(self, center_score_threshold, min_radius, max_radius, margin_threshold=0.1):
        # initialize a score map of potential circle centers
        center_score = np.zeros(self.value.shape[:2], dtype=int)
        indexes = np.array([
            [(y, x) for x in range(self.value.shape[1])] for y in range(self.value.shape[0])
        ])
        x_coords = indexes[:, :, 1]
        y_coords = indexes[:, :, 0]

        x_grad = gaussian(self.x_grad)
        y_grad = gaussian(self.y_grad)
        for y in range(self.value.shape[0]):
            for x in range(self.value.shape[1]):
                d_x = x_grad[y, x]
                d_y = y_grad[y, x]
                grad = np.hypot(d_x, d_y)
                # if it is an edge pixel
                if self.value[y, x]:
                    # all distance to each pixel in the image
                    x_distance = x_coords - x
                    y_distance = y_coords - y
                    distance = np.hypot(x_distance, y_distance)
                    # dx / x should equal to dy / y
                    # so, it is also equivalent to dx * y = dy * x
                    center_score[(np.abs(d_x * y_distance - d_y * x_distance) < margin_threshold) &
                                 (distance > min_radius) & (distance < max_radius)] += 1

        # non-maximum suppression
        for y in range(self.value.shape[0]):
            for x in range(self.value.shape[1]):
                neighbours = get_2darray_neighbours(center_score, row_index=y, column_index=x)
                # if the score is not the local maximum in the neighbours
                # set it as 0
                if not (center_score[y, x] >= neighbours).all():
                    center_score[y, x] = 0
        # record all locations of potential centers
        # each row is a center's location (y, x)
        potential_centers_indexes = np.vstack(np.where(center_score > center_score_threshold)).T
        edge_points_indexes = np.vstack(np.where(self.value)).T
        return ShapeCounter(center_score, centers_loc=potential_centers_indexes,
                            max_radius=max_radius, min_radius=min_radius, edge_points=edge_points_indexes)

    def sort_centers(self):
        def add_score(row):
            y, x = row
            return np.array([y, x, self.value[y, x]])

        # each row in centers: [y, x, score]
        centers = np.apply_along_axis(func1d=add_score, axis=1, arr=self.centers_loc)
        sorted_centers = centers[np.argsort(centers[:, 2])[::-1]]
        return ShapeCounter(sorted_centers, max_radius=self.max_radius, min_radius=self.min_radius,
                            edge_points=self.edge_points)

    def find_circles(self, center_distance_threshold, radius_score_threshold, radius_distance_threshold=5,
                     detect_concentric_circles=True):
        centers_result = []
        for i in range(self.value.shape[0]):
            terminate = False
            radius_score = np.zeros(self.max_radius + 1)
            # calculate the distance to all identified centers
            # if the center is too closed to another one
            for j in range(len(centers_result)):
                y_distance = self.value[i, 0] - centers_result[j][0]
                x_distance = self.value[i, 1] - centers_result[j][1]
                if (np.hypot(x_distance, y_distance) < center_distance_threshold).any():
                    # skip this center if there is another center nearby
                    terminate = True
                    break

            if terminate:
                continue
            # compute all edge points to this center point
            edge_points_distance = np.hypot((self.edge_points[:, 0] - self.value[i, 0]),
                                            (self.edge_points[:, 1] - self.value[i, 1]))
            # filtered the radius in the range
            edge_points_in_range = (edge_points_distance > self.min_radius) & (edge_points_distance < self.max_radius)
            # round the distance to int, for easier count scores
            edge_points_distance = np.round(edge_points_distance).astype(int)
            # count the radius score
            for j in range(len(edge_points_distance)):
                if edge_points_in_range[j]:
                    radius_score[edge_points_distance[j]] += 1
            # for each radius for this center point
            for radius in range(self.min_radius, len(radius_score)):
                # normalize the score by its radius. The more radius, the less the score.
                radius_score[radius] /= np.sqrt(radius)

            # descend order radius with radius scores
            sorted_radius = radius_score.argsort()[::-1]
            # filtered by the threshold
            sorted_radius = sorted_radius[np.where(radius_score[sorted_radius] > radius_score_threshold)]
            # if detect big circles
            if detect_concentric_circles:
                # remove the radius which are too closed to others
                filtered_radius = []
                for j, radius in enumerate(sorted_radius[::-1]):
                    # skip the largest radius
                    if j == len(sorted_radius):
                        continue
                    # the radius should have large gap to all other radius
                    if (np.abs(radius - sorted_radius[::-1][j + 1:]) > radius_distance_threshold).all():
                        filtered_radius.append(radius)

                # add the results to the center_results
                for radius in filtered_radius:
                    centers_result.append([self.value[i, 0], self.value[i, 1], radius])
            # if detect small spheres
            else:
                if len(sorted_radius) > 0:
                    centers_result.append([self.value[i, 0], self.value[i, 1], sorted_radius[0]])
        return ShapeCounter(np.array(centers_result))

    def add_circle_layers(self, circles, circle_thickness=1., center_thickness=5):
        # initialize the plot
        result = self.value.astype(int)
        circles = circles.value
        # iterate each pixel
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                # calculate the distance to each circle center point
                distance = np.hypot(circles[:, 0] - y, circles[:, 1] - x)
                # check if the pixel is in any circles' edges
                margin = np.abs(circles[:, 2] - distance)
                if (margin < circle_thickness).any():
                    # set the edge as yellow, which is like the example output
                    result[y, x] = (255, 255, 0)
                if (distance < center_thickness).any():
                    # set the center point as red
                    result[y, x] = (255, 0, 0)
        return ShapeCounter(result)


def get_2darray_neighbours(array, row_index, column_index):
    # return a 3x3 matrix which is the neighbour window for specified element
    row_high = min(row_index + 1, array.shape[0])
    row_low = max(0, row_index - 1)
    column_high = min(column_index + 1, array.shape[1])
    column_low = max(0, column_index - 1)
    return array[row_low:row_high + 1, column_low:column_high + 1]


if __name__ == '__main__':
    res = CountShapes(plt.imread(os.path.join("image", "ball-bearings-1958085_480.jpg")))
