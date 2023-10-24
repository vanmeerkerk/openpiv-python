""" Project images based on a pre-defined homography matrix

author: Mike van Meerkerk (Deltares)
date: 13-09-2023
"""
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Define the path to the homography matrix
    path_homography = Path(
        r"d:\11209221-012_LSPIV\03_BackProject\meting01\calibration\camera01_homography.csv"
    )
    # Define the camera matrix
    path_camera_matrix = Path(
        r"d:\11209221-012_LSPIV\03_BackProject\meting01\calibration\camera01_cameramatrix.csv"
    )
    # Path to the distortion coefficients
    path_distortion_coefficients = Path(
        r"d:\11209221-012_LSPIV\03_BackProject\meting01\calibration\camera01_distortion_coefficients.csv"
    )
    # Path to the image directory
    path_images = Path(
        r"p:\archivedprojects\11208914-meting-stuw-driel\HDD002\20230127-meting01"
    )
    # Path to the output directory
    path_output = Path(
        r"d:\11209221-012_LSPIV\03_BackProject\meting01\camera01\Faq_10Hz"
    )
    # Get the homography, and camera matrix and distortion coefficients
    homography_matrix = np.genfromtxt(path_homography, delimiter=",")
    camera_matrix = np.genfromtxt(path_camera_matrix, delimiter=",")
    distortion_coefficients = np.genfromtxt(path_distortion_coefficients, delimiter=",")
    # Reduce scale of homography matrix
    sx, sy = 0.1, 0.1
    homography_scale = np.eye(3)
    homography_scale[0, 0] = sx
    homography_scale[1, 1] = sy
    # Loop over N images in the image directory
    number_of_images = 1_000
    frame_skip = 5
    # Generate sorted list of file names
    image_name_list = sorted([file.stem for file in path_images.glob("*.tif")])

    for image_count in range(number_of_images):
        # Define the image file from the image generator
        image_file = Path(
            path_images, image_name_list[int(image_count * frame_skip)] + ".tif"
        )
        # Print statement
        print(f"Processing image {image_file.stem} ({image_count}/{number_of_images})")
        # Read the image
        image = cv2.cvtColor(cv2.imread(str(image_file)), cv2.COLOR_BGR2RGB)
        # Undistort image
        image_undistorted = cv2.undistort(image, camera_matrix, distortion_coefficients)
        # Project image
        image_projected = cv2.warpPerspective(
            image_undistorted,
            homography_scale @ homography_matrix,
            (int(10_000 * sx), int(10_000 * sy)),
            cv2.INTER_LINEAR,
        )
        # Save image
        image_output = Path(
            path_output, f"{image_count:05.0f}_projected_{image_file.stem}"
        ).with_suffix(image_file.suffix)
        cv2.imwrite(str(image_output), cv2.cvtColor(image_projected, cv2.COLOR_BGR2RGB))
        # # warp image corner points to determine bounding box
        # w, h, _ = image_undistorted.shape
        # points = [[0, 0], [0, h], [w, h], [w, 0]]
        # points = np.array(points, np.float32).reshape(-1, 1, 2)

        # warped_points = cv2.perspectiveTransform(points, homography_scale @ homography_matrix).squeeze()
