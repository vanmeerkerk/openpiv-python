"""Estimate the camera pose from a set of 2D-3D correspondences using the PnP algorithm.

author: Mike van Meerkerk (Deltares)
date: 11-08-2023
"""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Data file
    data_file = Path(
        r"n:\Projects\11208500\11208914\B. Measurements and calculations\12 - LS-PIV\01 - Markers\camera01_image_points_v02.xlsx"
    )

    # Load the 2D-3D correspondences from a CSV file
    df = pd.read_excel(data_file)

    # Extract the 2D and 3D coordinates from the dataframe
    image_points = df[["x", "y"]].values.astype(np.float32)[:-2, :]
    object_points = df[["X", "Y", "Z"]].values.astype(np.float32)[:-2, :]
    # subtract average from object points
    ref_point = np.array([183844.84, 441780.38, 13.46 - 6.59]).astype(np.float32)
    object_points = ref_point - object_points
    # Align coordinate sytem with the heading of the weir
    angle_weir = -62.5
    R_weir = cv.Rodrigues(np.array([0, 0, np.deg2rad(angle_weir)]))[0]
    object_points = object_points.dot(R_weir)

    # Camera 1
    camera_matrix = np.array(
        [
            [2727.8, 0, 1583.6],
            [0, 2723.7, 1081.0],
            [0, 0, 1],
        ]
    ).astype(np.float32)

    # Define the distortion coefficients
    distortion_coefficients = np.array([-0.0995, 0.2264, 0, 0, 0]).astype(np.float32)

    # # Camera 2
    # camera_matrix = np.array(
    #     [
    #         [2730.6, 0, 1598.8],
    #         [0, 2726.9, 1065.5],
    #         [0, 0, 1],
    #     ]
    # )

    # # Define the distortion coefficients
    # distortion_coefficients = np.array([-0.0922, 0.1874, 0, 0, 0])
    # Estimate the camera pose using the PnP algorithm
    retval, rotation_vector, translation_vector = cv.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv.SOLVEPNP_ITERATIVE,
    )

    # Check backprojection error: we do not obtain an accurate back-projection result. The method in itself seems to be correct!
    back_proj_image_points, _ = cv.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients,
    )
    back_proj_image_points = np.squeeze(back_proj_image_points)
    error_back_proj = np.linalg.norm((back_proj_image_points - image_points), axis=1)

    # Plot image points
    plt.figure()
    plt.imshow(image)
    plt.plot(image_points[:, 0], image_points[:, 1], "og", markersize=5)
    plt.plot(
        back_proj_image_points[:, 0], back_proj_image_points[:, 1], "+r", markersize=15
    )
    plt.show(block=False)

    # Print the estimated rotation and translation vectors
    print(f"Rotation vector: {rotation_vector}")
    print(f"Translation vector: {translation_vector}")

    # Dewarp the image
    image_name = Path(
        r"p:/archivedprojects/11208914-meting-stuw-driel/HDD001/20230127-meting01/093453.661.tif"
    )

    image = cv.cvtColor(cv.imread(str(image_name)), cv.COLOR_BGR2RGB)
    # Get image shape
    image_height, image_width, _ = image.shape
    # Define the new camera matrix with alpha=1 all pixels are retained for alpha=0 only valid pixels are retained
    alpha = 1
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coefficients, (image_width, image_height), alpha
    )
    # undistort
    image_undistorted = cv.undistort(
        image,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix,
    )

    plt.figure()
    plt.imshow(image_undistorted)

    # Rotation vector to rotation matrix
    R, _ = cv.Rodrigues(rotation_vector)

    # Open-earth rotation matrix
    z_shift = -(13.46 - 6.59)
    R_open_earth = np.hstack(
        [
            R[:, :2],
            np.expand_dims(R[:, 2] * z_shift + translation_vector.flatten(), 1),
        ]
    )
    # Compute homography matrix
    H = np.linalg.inv(np.matmul(new_camera_matrix, R_open_earth))
    # Normalize homography matrix
    H = H / H[2, 2]

    # warp image corner points to determine bounding box
    w, h, _ = image.shape
    points = [[0, 0], [0, h], [w, h], [w, 0]]
    points = np.array(points, np.float32).reshape(-1, 1, 2)

    warped_points = cv.perspectiveTransform(points, H).squeeze()

    # Determine bounding box
    bbox = cv.boundingRect(warped_points.astype(np.int32))

    # Apply offset to homography matrix
    H_shift = np.eye(3)
    H_shift[:2, 2] = np.array([-bbox[0], -bbox[1]])

    # Calculate new warped points and bounding box
    warped_points_shift = cv.perspectiveTransform(
        points, np.matmul(H_shift, H)
    ).squeeze()
    bbox_shift = cv.boundingRect(warped_points_shift.astype(np.int32))

    # Determine the scaling homography
    sx, sy = 100, 100
    H_scale = np.eye(3)
    H_scale[0, 0] = sx
    H_scale[1, 1] = sy

    # Reference point to change the origin of the image
    reference_point = np.matmul(H_shift, H).dot(
        np.vstack([np.expand_dims(image_points[6, :], axis=1), 1])
    )
    reference_point = reference_point / reference_point[2]

    # Determine extend of the warpPerspective image based on x- and y-limits of the unwarped domain
    x_domain_size = 120 * sx
    x_limit_array = x_domain_size * np.array([-0.5, 0.5])
    x_domain_min, x_domain_max = (x_limit_array + reference_point[0]).astype(int)

    y_limit_array = np.array([-5, 200]) * sy
    y_domain_min, y_domain_max = (y_limit_array - reference_point[1]).astype(int)

    # Determine extend of warpPerspective
    x_domain_size = x_domain_max - x_domain_min
    y_domain_size = y_domain_max - y_domain_min

    # Detemine additional shift homography to center on reference point
    H_shift_origin = np.eye(3)
    H_shift_origin[:2, 2] = np.array([-x_domain_min, -y_domain_min])

    image_warped = cv.warpPerspective(
        image_undistorted,
        H_shift_origin @ H_shift @ H_scale @ H,
        (x_domain_size, y_domain_size),
        cv.INTER_NEAREST,
    )

    # # Plot image
    plt.figure()
    plt.imshow(
        image_warped,
        extent=(x_limit_array[0], x_limit_array[1], y_limit_array[1], y_limit_array[0]),
    )
    plt.gca().invert_yaxis()
    plt.grid(color=[1, 1, 1, 0.2])
    plt.xlim(x_limit_array)
    plt.ylim(y_limit_array)
    plt.xlabel(r"$x$ [cm]")
    plt.ylabel(r"$y$ [cm]")
