import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy import stats
from math import atan, sin, cos
from scipy.signal import find_peaks


def hough_transform(points):

    centre = np.mean(points, axis=0)
    points_num = points.shape[0]
    row_transformed_data = int(points_num * (points_num - 1) / 2)
    hough_space = np.zeros(shape=(row_transformed_data, 4))

    for i in range(points_num):
        for j in range(i + 1, points_num):
            idx = int(i * points_num - i * (i + 1) / 2 + j - i - 1)

            x_i, y_i = points[i] - centre
            x_j, y_j = points[j] - centre

            theta = atan((x_j - x_i) / (y_i - y_j))

            rho = x_i * cos(theta) + y_i * sin(theta)

            hough_space[idx][0] = theta
            hough_space[idx][1] = rho
            hough_space[idx][2] = i
            hough_space[idx][3] = j

    return hough_space


def find_kde_peaks(data, bw=0.05):
    """
    returns: m x 2 NDarray \\
    column 0: value from data corrsponding to the peak_kde_values \\
    column 1: peak kde values
    """

    kde = stats.gaussian_kde(data, bw_method=bw)
    x_range_min = min(data) - 0.1 * abs(min(data))
    x_range_max = max(data) + 0.1 * abs(max(data))

    x_range = np.linspace(x_range_min, x_range_max, 1000)
    kde_values = kde(x_range)

    peaks, _ = find_peaks(kde_values)

    x_peaks = x_range[peaks].reshape(-1, 1)
    kde_values_peaks = kde_values[peaks].reshape(-1, 1)

    kde_peaks = np.hstack((x_peaks, kde_values_peaks))

    return kde_peaks


def find_top_two_kde_peaks(kde_peaks):

    sorted_indices = np.argsort(kde_peaks[:, 1])[::-1]
    top_two_indices = sorted_indices[:2]
    top_two_kde_peaks = kde_peaks[top_two_indices]

    return top_two_kde_peaks


def transform_along_basis(points, basis_i, basis_j):

    transformed_points = []

    for point in points:
        # 计算点在基向量上的投影系数
        coeff_i = np.dot(point, basis_i) / np.dot(basis_i, basis_i)
        coeff_j = np.dot(point, basis_j) / np.dot(basis_j, basis_j)
        transformed_points.append((coeff_i, coeff_j))

    return np.array(transformed_points)


def calculate_nearest_location(points, grids_x_axes, grids_y_axes):

    nearest_grids = []

    for pt in points:
        x_min_idx = (np.square(grids_x_axes - pt[0])).argmin()
        y_min_idx = (np.square(grids_y_axes - pt[1])).argmin()

        nearest_grids.append((x_min_idx, y_min_idx))

    nearest_grids = np.array(nearest_grids)

    while True:

        unique_grids, counts = np.unique(nearest_grids, axis=0, return_counts=True)
        repeated_grids = unique_grids[counts > 1]

        if len(repeated_grids) == 0:
            break
        for rep_grid in repeated_grids:

            indices = np.where((nearest_grids == rep_grid).all(axis=1))[0]
            # change the top two
            idx_pt_a, idx_pt_b = indices[0], indices[1]

            idx_grid_i, idx_grid_j = rep_grid[0], rep_grid[1]

            # points: 2x1 array
            pt_a, pt_b = points[idx_pt_a], points[idx_pt_b]
            # index: int
            grid_i, grid_j = grids_x_axes[idx_grid_i], grids_y_axes[idx_grid_j]

            # compare the x values of pt_a and pt_b
            # change the less one's nearest grid
            # change rule:
            #   compare the dist along the i_axis and j_axis,
            #   move 1 step along the axis whose dist is greater
            if pt_a[0] < pt_b[0]:

                # move pt_b
                x_to_grid_dist = np.square(pt_b[0] - grid_i)
                y_to_grid_dist = np.square(pt_b[1] - grid_j)

                if x_to_grid_dist < y_to_grid_dist:
                    new_nearest_grid = [idx_grid_i, idx_grid_j + 1]
                else:
                    new_nearest_grid = [idx_grid_i + 1, idx_grid_j]
                # update the nearest_grids
                nearest_grids[idx_pt_b] = np.array(new_nearest_grid)
            else:

                # move pt_a
                x_to_grid_dist = np.square(pt_a[0] - grid_i)
                y_to_grid_dist = np.square(pt_a[1] - grid_j)

                if x_to_grid_dist < y_to_grid_dist:
                    new_nearest_grid = [idx_grid_i, idx_grid_j + 1]
                else:
                    new_nearest_grid = [idx_grid_i + 1, idx_grid_j]
                # update the nearest_grids
                nearest_grids[idx_pt_a] = np.array(new_nearest_grid)

    return nearest_grids


# visualization


if __name__ == "__main__":

    points = np.genfromtxt("./soucefile/coords.csv", delimiter=",")

    # step-1: finding the top two theta
    hough_space = hough_transform(points)

    hough_space_theta = hough_space[:, 0]
    hough_space_rho = hough_space[:, 1]

    theta_kde_peaks = find_kde_peaks(hough_space_theta)
    top_two_kde_peaks = find_top_two_kde_peaks(theta_kde_peaks)

    theta_1 = top_two_kde_peaks[0, 0]
    theta_2 = top_two_kde_peaks[1, 0]

    # step-2: transform the orginal data along the new basis
    basis_i = np.array([cos(theta_1), sin(theta_1)])
    basis_j = np.array([cos(theta_2), sin(theta_2)])
    centre = np.mean(points, axis=0)
    points_centered = points - centre
    transformed_points = transform_along_basis(points_centered, basis_i, basis_j)

    # step-3: finding the grid
    grids_i_axes = find_kde_peaks(transformed_points[:, 0], bw=0.07)[:, 0]

    grid_j_axes = find_kde_peaks(transformed_points[:, 1], bw=0.07)[:, 0]

    # step-4: set the points into the grid
    nearest_grids = calculate_nearest_location(
        transformed_points, grids_i_axes, grid_j_axes
    )
