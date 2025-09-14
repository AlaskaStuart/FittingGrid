import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def find_centres(values, number_of_centres: int):
    kmeans = KMeans(n_clusters=number_of_centres)
    # Reshape because this needs a dummy dimension.
    kmeans.fit(values.reshape(-1, 1))
    return np.array(kmeans.cluster_centers_)


def rotate_points(points, angle):
    """Rotate the points around the origin by `angle` radians.
    Modified from
    https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    xs, ys = points[0, :], points[1, :]
    rotated_xs = np.cos(angle) * xs - np.sin(angle) * ys
    rotated_ys = np.sin(angle) * xs + np.cos(angle) * ys
    return rotated_xs, rotated_ys


def display_found_grid(points, res_xs, res_ys, angle, vertical_lines, horizontal_lines):
    fig, ax = plt.subplots()
    xs, ys = points[0, :], points[1, :]
    ax.scatter(xs, ys)

    for x_centre in np.array(res_xs).T:
        ax.axline(x_centre, slope=1 / angle, color="black", alpha=0.6)

    for y_centre in np.array(res_ys).T:
        ax.axline(y_centre, slope=-angle, color="black", alpha=0.6)

    ax.set_title(
        f"Fitting a non-uniform ({horizontal_lines}x{vertical_lines}) grid to points\n"
        + f"using a fixed angle of: {angle:0.3} (radians)"
    )
    ax.set_aspect("equal")

    plt.show()


def fit_grid_to_points(
    points, angle: float, vertical_lines: int, horizontal_lines: int
):
    rotated_xs, rotated_ys = rotate_points(points, angle)

    x_centres = find_centres(rotated_xs, vertical_lines)
    y_centres = find_centres(rotated_ys, horizontal_lines)

    # Rotate the centres back. Match up with the original alignment.
    x_centre_points = np.hstack((x_centres, np.zeros_like(x_centres))).T
    y_centre_points = np.hstack((np.zeros_like(y_centres), y_centres)).T

    rotated_back_x_centres = rotate_points(x_centre_points, -angle)
    rotated_back_y_centres = rotate_points(y_centre_points, -angle)

    return rotated_back_x_centres, rotated_back_y_centres


def main():
    pts = np.array(
        [
            (104, 131),
            (240, 136),
            (580, 183),
            (88, 234),
            (396, 277),
            (199, 431),
            (367, 451),
            (534, 464),
            (29, 554),
            (171, 627),
            (342, 628),
            (493, 638),
            (10, 739),
            (144, 747),
            (138, 927),
            (472, 966),
        ]
    ).T

    angle = -0.14  # radians
    vertical_lines, horizontal_lines = 4, 6

    res_xs, res_ys = fit_grid_to_points(pts, angle, vertical_lines, horizontal_lines)
    display_found_grid(pts, res_xs, res_ys, angle, vertical_lines, horizontal_lines)


if __name__ == "__main__":
    main()
