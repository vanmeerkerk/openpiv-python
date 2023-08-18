"""Retrieve red dots from calibration images.

This script retrieves the red dots from the calibration images and saves them
in a .txt file. The .txt file is used in the calibration script to determine
the camera extrinsic parameters. The red dots are retrieved by using the 
OpenCV function

author: Mike van Meerkerk (Deltares)
date: 03 August 2023
"""
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# %% Import
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# %% Retrieve the red dots from the calibration images
if __name__ == "__main__":
    #     # Define the path to the calibration images
    #     calib_folder = (
    #         r"p:\archivedprojects\11208914-meting-stuw-driel\HDD001\20230127-meting01"
    #     )
    #     # Retrieve the calibration images
    #     calib_images = Path(calib_folder).glob("*.tif")

    #     # Loop over the calibration images to retrieve the red dots
    #     # for image_name in calib_images:
    #     #     # Open image file
    #     #     image = cv.imread(str(image_name))

    # # Read example
    # # Camera 1
    # image_name = Path(
    #     r"p:/archivedprojects/11208914-meting-stuw-driel/HDD001/20230127-meting01/093812.040.tif"
    # )
    # # Camera 2
    # image_name = Path(
    #     r"p:/archivedprojects/11208914-meting-stuw-driel/HDD002/20230127-meting01/093812.037.tif"
    # )

    # image = cv.cvtColor(cv.imread(str(image_name)), cv.COLOR_BGR2RGB)
    # # Red markers
    # # Get marker 1
    # marker1 = image[slice(220, 250), slice(210, 240), :]
    # # Get marker 2
    # marker2 = image[slice(190, 220), slice(860, 890), :]
    # # Get marker 3
    # marker3 = image[slice(100, 130), slice(1290, 1320), :]
    # # Yellow reference on pillar
    # marker4 = image[slice(1740, 1840), slice(1080, 1180), :]

    # %%
    # Get images in path
    image_folder = Path(
        r"p:/archivedprojects/11208914-meting-stuw-driel/HDD001/20230127-meting01"
    )
    image_list = list(image_folder.glob("*.tif"))

    # Average n-images in path
    n = 60
    alpha = 1 / n
    initialized_image = False
    # Define function to track average of red color channel in image
    slice_lst = [slice(225, 245), slice(215, 235)]
    run_avg_red = np.zeros((1, n))
    # Loop
    for image_index, image_name in enumerate(image_list[:n]):
        n_avg = image_index + 1
        print(f"Processing image {image_index + 1} of {n}")

        # Src image
        src_image = cv.imread(str(image_name))
        # 2D median filter to remove noise while preserving edges
        src_image = cv.medianBlur(src_image, 3)
        # Initialize average image
        if not initialized_image:
            avg_image = src_image
            initialized_image = True
        elif image_index < n:
            avg_image = avg_image * (n_avg - 1) / n_avg + src_image * (1 / n_avg)
        else:
            avg_image = (1 - alpha) * avg_image + alpha * src_image
        # Average red color channel
        run_avg_red[0, image_index] = np.mean(avg_image[slice_lst[0], slice_lst[1], 2])
    # Convert image to uint8 and to RGB color space.
    avg_image = cv.cvtColor(np.uint8(avg_image), cv.COLOR_BGR2RGB)
    # Plot average
    # plt.figure()
    # plt.imshow(avg_image)

    # # %% Plot sliced 3D surface
    # from mpl_toolkits import mplot3d

    # # Window size
    # window_size = 50
    # # Marker 6
    # xc_estimate, yc_estimate = (226, 234)
    # # Marker 5
    # # xc_estimate, yc_estimate = (875, 204)

    # # Slice list based on window size and center estimate
    # slice_lst = [
    #     slice(yc_estimate - window_size // 2, yc_estimate + window_size // 2),
    #     slice(xc_estimate - window_size // 2, xc_estimate + window_size // 2),
    # ]

    # # Mesh grid
    # x, y = np.meshgrid(
    #     np.arange(slice_lst[1].start, slice_lst[1].stop),
    #     np.arange(slice_lst[0].start, slice_lst[0].stop),
    # )

    # z = avg_image[slice_lst[0], slice_lst[1], 0]
    # # Maximum of red color channel
    # indx_max_red = np.argmax(z)
    # xc, yc = [x.ravel()[indx_max_red], y.ravel()[indx_max_red]]
    # # Plot 3D surface of the sliced average image

    # fig = plt.figure(figsize=(10, 5))
    # # Set title
    # fig.suptitle("Marker 1: 2D image and 3D surface")
    # # Plot image
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(
    #     avg_image[slice_lst[0], slice_lst[1], :],
    #     extent=[x.min(), x.max(), y.min(), y.max()],
    #     vmin=0,
    #     vmax=255,
    # )
    # ax.set_xlabel("x [px]")
    # ax.set_ylabel("y [px]")
    # # Set 3D projection to 2nd axis
    # switch_3d = True

    # if switch_3d:
    #     # Plot 3D surface
    #     axs = fig.add_subplot(1, 2, 2, projection="3d")
    #     axs.plot_surface(x, y, z, cmap="jet")
    #     axs.set_xlabel("x [px]")
    #     axs.set_ylabel("y [px]")
    #     axs.set_zlabel("red color channel")
    # else:
    #     axs = fig.add_subplot(1, 2, 2)
    #     plt.pcolormesh(
    #         x, y, z, cmap="jet", vmin=z.min(), vmax=z.max(), shading="auto", alpha=1
    #     )
    #     axs.plot([x.min(), x.max()], [232.64] * 2, "-r")
    #     axs.plot([224.6] * 2, [y.min(), y.max()], "-r")
    #     axs.set_xlabel("x [px]")
    #     axs.set_ylabel("y [px]")
    # # axs.plot([x.min(), x.max()], [106.62, 106.62], "-r")
    # # axs.plot([1303.62, 1303.62], [y.min(), y.max()], "-r")

    # from scipy.constants import *
    # from scipy.integrate import simps

    # # %% HSV filtering

    # # Convert BGR to HSV
    # hsv = cv.cvtColor(avg_image, cv.COLOR_RGB2HSV)
    # # Select in range for red_1 and red_2
    # lower_red_1 = np.array([0, 50, 100])
    # upper_red_1 = np.array([20, 255, 255])
    # lower_red_2 = np.array([160, 50, 100])
    # upper_red_2 = np.array([180, 255, 255])

    # # Threshold the HSV image to get only red colors
    # mask_1 = cv.inRange(hsv, lower_red_1, upper_red_1)
    # mask_2 = cv.inRange(hsv, lower_red_2, upper_red_2)

    # mask = mask_1 + mask_2

    # # Bitwise-AND mask and original image
    # res = cv.bitwise_and(avg_image, avg_image, mask=mask)
    # # Plot
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(hsv)
    # plt.subplot(222, sharex=plt.subplot(221), sharey=plt.subplot(221))
    # plt.imshow(mask)
    # plt.subplot(223, sharex=plt.subplot(221), sharey=plt.subplot(221))
    # plt.imshow(res)
    # plt.subplot(224, sharex=plt.subplot(221), sharey=plt.subplot(221))
    # plt.imshow(avg_image)

    # from scipy.interpolate import RegularGridInterpolator

    # # %% Fit 2D gaussian
    # from scipy.stats import multivariate_normal

    # def integrate_simps(mesh, func):
    #     nx, ny = func.shape
    #     px, py = mesh[0][int(nx / 2), :], mesh[1][:, int(ny / 2)]
    #     val = simps(simps(func, px), py)
    #     return val

    # def normalize_integrate(mesh, func):
    #     return func / integrate_simps(mesh, func)

    # def moment(mesh, func, index):
    #     ix, iy = index[0], index[1]
    #     g_func = normalize_integrate(mesh, func)
    #     fxy = g_func * mesh[0] ** ix * mesh[1] ** iy
    #     val = integrate_simps(mesh, fxy)
    #     return val

    # def moment_seq(mesh, func, num):
    #     seq = np.empty([num, num])
    #     for ix in range(num):
    #         for iy in range(num):
    #             seq[ix, iy] = moment(mesh, func, [ix, iy])
    #     return seq

    # def get_centroid(mesh, func):
    #     dx = moment(mesh, func, (1, 0))
    #     dy = moment(mesh, func, (0, 1))
    #     return dx, dy

    # def get_weight(mesh, func, dxy):
    #     g_mesh = [mesh[0] - dxy[0], mesh[1] - dxy[1]]
    #     lx = moment(g_mesh, func, (2, 0))
    #     ly = moment(g_mesh, func, (0, 2))
    #     return np.sqrt(lx), np.sqrt(ly)

    # def get_covariance(mesh, func, dxy):
    #     g_mesh = [mesh[0] - dxy[0], mesh[1] - dxy[1]]
    #     Mxx = moment(g_mesh, func, (2, 0))
    #     Myy = moment(g_mesh, func, (0, 2))
    #     Mxy = moment(g_mesh, func, (1, 1))
    #     return np.array([[Mxx, Mxy], [Mxy, Myy]])

    # mesh = [x, y]

    # s0xy = get_centroid(mesh, z)
    # w0xy = get_covariance(mesh, z, s0xy)
    # fxy1 = multivariate_normal.pdf(np.stack(mesh, -1), mean=s0xy, cov=w0xy)

    # from linecache import clearcache, getline

    # from mpl_toolkits.axes_grid1 import make_axes_locatable

    # def plot_contour_sub(mesh, func, loc=[0, 0], title="name", pngfile="./name"):
    #     sx, sy = loc
    #     nx, ny = func.shape
    #     xs, ys = mesh[0][0, 0], mesh[1][0, 0]
    #     dx, dy = mesh[0][0, 1] - mesh[0][0, 0], mesh[1][1, 0] - mesh[1][0, 0]
    #     mx, my = int((sy - ys) / dy), int((sx - xs) / dx)
    #     fig, ax = plt.subplots()
    #     divider = make_axes_locatable(ax)
    #     ax.set_aspect("equal")
    #     ax_x = divider.append_axes("bottom", 1.0, pad=0.5, sharex=ax)
    #     ax_x.plot(mesh[0][mx, :], func[mx, :])
    #     ax_x.set_title("y = {:.2f}".format(sy))
    #     ax_y = divider.append_axes("right", 1.0, pad=0.5, sharey=ax)
    #     ax_y.plot(func[:, my], mesh[1][:, my])
    #     ax_y.set_title("x = {:.2f}".format(sx))
    #     im = ax.contourf(*mesh, func, cmap="jet")
    #     ax.set_title(title)
    #     plt.colorbar(im, ax=ax, shrink=0.9)

    # plot_contour_sub(mesh, fxy1, loc=s0xy, title="Reconst")
    # # Add data to plots
    # interp_object = RegularGridInterpolator((x[0, :], y[:, 0]), z)
    # x_interp = interp_object(np.c_[x[0, :], np.ones_like(x[0, :]) * s0xy[1]])
    # # %% Plot image and manually get markers
    # # source: https://stackoverflow.com/questions/29145821/can-python-matplotlib-ginput-be-independent-from-zoom-to-rectangle

    # plt.figure()
    # plt.imshow(
    #     cv.cvtColor(cv.medianBlur(image, 3), cv.COLOR_RGB2GRAY),
    #     cmap="gray",
    #     vmin=0,
    #     vmax=255,
    # )
    # Get n-points from the image
    plt.figure()
    plt.imshow(avg_image)
    n = 8
    point_lst = plt.ginput(
        n=0, timeout=0, mouse_add=None, mouse_pop=None, mouse_stop=None
    )

    # Convert point_lst to numpy array
    points = np.array(point_lst)

    # Create dict to save
    # Camera 1
    # markers = {
    #     "marker4_center": points[0, :],
    #     "marker3_center": points[1, :],
    #     "marker2_center": points[2, :],
    #     "marker1_center": points[3, :],
    #     "marker_pillar_left": points[4, :],
    #     "marker_pillar_center": points[5, :],
    #     "marker_grote_paal_rechts": points[6, :],
    #     "marker_witte_paal_rechts": points[7, :],
    # }
    # Camera 2
    markers = {f"marker{i+1}_center": el for i, el in enumerate(points)}
    # Save the markers to an excel file
    import pandas as pd

    df = pd.DataFrame.from_dict(markers, orient="index")
    df.columns = ["x", "y"]
    df.to_excel(
        r"n:\Projects\11208500\11208914\B. Measurements and calculations\12 - LS-PIV\01 - Markers\camera01_image_points_v02.xlsx"
    )
    # Plot the markers with text in the image
    plt.figure()
    plt.imshow(avg_image)
    for key, value in markers.items():
        plt.plot(value[0], value[1], "r+", markersize=15)
        plt.text(value[0], value[1], key, color="r")

    # %% Hough transform
    # circles = cv.HoughCircles(marker1[:, :, 0], cv.HOUGH_GRADIENT, 1.2, 100)
    # # %% Detect dots
    # # Set our filtering parameters
    # # Initialize parameter setting using cv2.SimpleBlobDetector
    # params = cv.SimpleBlobDetector_Params()

    # # Set Area filtering parameters
    # params.filterByArea = True
    # params.minArea = 10

    # # # Set Circularity filtering parameters
    # # params.filterByCircularity = True
    # # params.minCircularity = 0.9

    # # # Set Convexity filtering parameters
    # # params.filterByConvexity = True
    # # params.minConvexity = 0.2

    # # # Set inertia filtering parameters
    # # params.filterByInertia = True
    # # params.minInertiaRatio = 0.01

    # # Create a detector with the parameters
    # detector = cv.SimpleBlobDetector_create(params)

    # # Detect blobs
    # keypoints = detector.detect(marker1[:, :, 0])

    # # %% Plot the markers in a single figure with only a single color channel
    # # Plot the markers
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(marker1[:, :, 0] / marker1.max(), cmap="gray", vmin=0, vmax=1)
    # plt.subplot(222)
    # plt.imshow(marker2[:, :, 0] / marker1.max(), cmap="gray", vmin=0, vmax=1)
    # plt.subplot(223)
    # plt.imshow(marker3[:, :, 0] / marker1.max(), cmap="gray", vmin=0, vmax=1)
    # # %%
