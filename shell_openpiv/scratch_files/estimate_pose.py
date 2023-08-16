"""Estimate the camera pose from a set of 2D-3D correspondences using the PnP algorithm.

author: Mike van Meerkerk (Deltares)
date: 11-08-2023
"""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_pixel_coordinates(img):
    """Get pixel coordinates given an image

    Parameters
    ----------
    img : np.ndarray
        NxMx1 or NxMx3 image matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing u-coordinates
    np.ndarray
        NxM matrix containing v-coordinates
    """

    # get pixel coordinates
    U, V = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    return U, V


def rectify_coordinates(U, V, H):
    """Get projection of image pixels in real-world coordinates
    given image coordinate matrices and  homography

    Parameters
    ----------
    U : np.ndarray
        NxM matrix containing u-coordinates
    V : np.ndarray
        NxM matrix containing v-coordinates
    H : np.ndarray
        3x3 homography matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing real-world x-coordinates
    np.ndarray
        NxM matrix containing real-world y-coordinates
    """

    UV = np.vstack((U.flatten(), V.flatten())).T

    # transform image using homography
    XY = cv.perspectiveTransform(np.asarray([UV]).astype(np.float32), H)[0]

    # reshape pixel coordinates back to image size
    X = XY[:, 0].reshape(U.shape[:2])
    Y = XY[:, 1].reshape(V.shape[:2])

    return X, Y


def rectify_image(img, H):
    """Get projection of image pixels in real-world coordinates
    given an image and homography

    Parameters
    ----------
    img : np.ndarray
        NxMx1 or NxMx3 image matrix
    H : np.ndarray
        3x3 homography matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing real-world x-coordinates
    np.ndarray
        NxM matrix containing real-world y-coordinates
    """

    U, V = get_pixel_coordinates(img)
    X, Y = rectify_coordinates(U, V, H)

    return X, Y


def draw_axis(
    img,
    R,
    t,
    K,
    length_of_axis=100,
    offset_point=(183803.158, 441756.219, 10.267),
    thickness=5,
):
    # unit is mm
    rotV, _ = cv.Rodrigues(R)
    # Define points to project on the image
    points = np.float32(
        [
            [length_of_axis, 0, 0],
            [0, length_of_axis, 0],
            [0, 0, length_of_axis],
            [0, 0, 0],
        ]
    )
    # Add offset
    points = points + np.array(offset_point)

    # Reshape points
    points = points.reshape(-1, 3)
    # Project points
    axisPoints, _ = cv.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    # Plot lines from the center to the projected points
    corner = tuple(axisPoints[-1].ravel().astype(int))
    img = cv.line(
        img, corner, tuple(axisPoints[0].ravel().astype(int)), (255, 0, 0), thickness
    )
    img = cv.line(
        img, corner, tuple(axisPoints[1].ravel().astype(int)), (0, 255, 0), thickness
    )
    img = cv.line(
        img, corner, tuple(axisPoints[2].ravel().astype(int)), (0, 0, 255), thickness
    )
    return img


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
    # P = np.matmul(camera_matrix, np.hstack([cv.Rodrigues(rotation_vector)[0], translation_vector]))
    # homo_object_points = (np.hstack([object_points, np.ones((object_points.shape[0], 1))])).T
    # back_proj_image_points = np.matmul(P, homo_object_points)
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
    # Determine offset with the inverse of the homography matrix on the image (0, 0) point in homogeneous coordinates
    # x_offset, y_offset, _ = -np.linalg.inv(H).dot(np.array([0, 0, 1]))
    # x_offset, y_offset = 0, 0  # 31.1
    # x_offset, y_offset = np.array(image.shape[:2]) // 2
    # # Apply offset to homography matrix
    # # x_offset, y_offset = new_camera_matrix[:2, 2].astype(np.int32)

    # H_shift = np.eye(3)
    # H_shift[:2, 2] = np.array([x_offset, y_offset])
    # Offseted homography matrix
    # H = H_shift.dot(H)
    # Normalize homography matrix
    H = H / H[2, 2]

    # warp image corner points:
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

    # # Rotate image about center see https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=getperspective#getrotationmatrix2d
    # image_warped = cv.warpPerspective(
    #     image_undistorted,
    #     np.matmul(H_shift, H),
    #     (bbox_shift[2], bbox_shift[3]),
    #     cv.INTER_NEAREST,
    # )
    # Rotate
    # from scipy import ndimage

    # angle = -82.5  # (90 - 29.16761337957779)
    # # Angel from direction of driel
    # angle = -62.5

    # image_warped_and_rotated = ndimage.rotate(image_warped, angle, reshape=True)
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

    y_limit_array = np.array([-10, 200]) * sy
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
#     # get size and offset of warped corner points:
#     xmin, ymin = warped_points.min(axis=0)
#     xmax, ymax = warped_points.max(axis=0)

#     # size:
#     warped_image_size = int(round(xmax - xmin)), int(round(ymax - ymin))

#     # warp image
#     image_warped = cv.warpPerspective(
#         image_undistorted, H, warped_image_size, cv.INTER_CUBIC
#     )
#     # Rotate image about center see https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=getperspective#getrotationmatrix2d
#     # # Plot image
#     plt.figure()
#     plt.imshow(image_warped)
#     plt.gca().invert_yaxis()

#     # warped_image_size = int(round(X.max()-X.min())), int(round(Y.max()-Y.min()))
#     # # # offset:
#     # # Ht = np.eye(3)
#     # # Ht[0, 2] = -xmin
#     # # Ht[1, 2] = -ymin

#     # # H = Ht @ Hr

#     # test = 0
#     # # # Draw the axis on the image
#     # # image_axis = draw_axis(
#     # #     image_undistorted,
#     # #     rotation_vector,
#     # #     translation_vector,
#     # #     camera_matrix,
#     # #     length_of_axis=5e3,
#     # # )
#     # # plt.figure()
#     # # plt.imshow(image_axis)

# # # Camera homogoraphy matrix
# # P = new_camera_matrix.dot(np.hstack((R, translation_vector)))
# # # Camera matrix times rotation matrix
# # M = new_camera_matrix.dot(R)

# # # Center based on homogenous coordinates (see page 163 of Hartley and Zisserman)
# # center = np.hstack(
# #     (
# #         np.linalg.det(P[:, 1:]),
# #         -np.linalg.det(P[:, [0, 2, 3]]),
# #         np.linalg.det(P[:, [0, 1, 3]]),
# #         -np.linalg.det(P[:, [0, 1, 2]]),
# #     )
# # )
# # # See https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
# # # Determine camera center (see equation 6.8 of Hartley and Zisserman)
# # camera_center = -np.linalg.inv(R).dot(translation_vector)
# # # Distance from camera center to objective plane
# # distance = np.abs(camera_center[2])[0]
# # # Determine the rotation matrix between the two planes assuming that the objective plane is parallel to the x-y plane
# # R_plane = np.eye(3).dot(R.transpose())
# # # Determine the translation vector between the two planes
# # t_plane = np.eye(3).dot(-R.transpose().dot(translation_vector)) + (
# #     translation_vector + np.array([[0], [0], [-distance]])
# # )
# # # Determine the normal
# # n_plane = np.array([[0], [0], [1]])
# # # Determine the homography matrix
# # H = R_plane + (t_plane.dot(n_plane.transpose())) / distance

# # # # C = -np.linalg.inv(R)*translation_vector
# # # # Ht = -camera_matrix * C
# # # # Hr = new_camera_matrix * np.linalg.inv(R) * np.linalg.inv(camera_matrix)
# # # R, _ = cv.Rodrigues(rotation_vector)
# # # Hr = camera_matrix @ R.T @ np.linalg.pinv(camera_matrix)
