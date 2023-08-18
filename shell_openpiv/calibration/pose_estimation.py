"""Estimate the pose of the camera using a world coordinates

author: Mike van Meerkerk (Deltares)
date: 01 August 2023
"""
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

from typing import List, NamedTuple, Optional

import cv2 as cv
import numpy as np


class Distortion(NamedTuple):
    """Distortion coefficients"""

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    # Class method to output distortion coefficients as an array
    @classmethod
    def as_array(cls) -> np.ndarray:
        """Distortion coefficients as array"""
        return np.array([cls.k1, cls.k2, cls.p1, cls.p2, cls.k3])


class CameraMatrix(NamedTuple):
    """Camera matrix"""

    fx: float
    fy: float
    cx: float
    cy: float

    # Class method to output camera matrix as an array
    @classmethod
    def as_array(cls) -> np.ndarray:
        """Camera matrix as array"""
        return np.array([[cls.fx, 0, cls.cx], [0, cls.fy, cls.cy], [0, 0, 1]])


class RotationMatrix(NamedTuple):
    """Rotation matrix"""

    r11: float
    r12: float
    r13: float
    r21: float
    r22: float
    r23: float
    r31: float
    r32: float
    r33: float


class TranslationVector(NamedTuple):
    """Translation vector"""

    t1: float
    t2: float
    t3: float


class Camera:
    """Camera class

    The camera class contains the camera parameters and the estimated pose.

    Parameters:
    ----------
    dist : list
        Distortion coefficients [k1, k2, p1, p2, k3]
    mtx : list
        Camera matrix [fx, fy, cx, cy]
    rvec : list, optional
        Rotation vector [r1, r2, r3]
    tvec : list, optional
        Translation vector [t1, t2, t3]
    world_points : list, optional
        World points [[x1, y1, z1], [x2, y2, z2], ...]
    image_points : list, optional
        Image points [[x1, y1], [x2, y2], ...]
    """

    def __init__(
        self,
        dist: List[float],
        mtx: List[float],
        rvec: Optional[List[float]] = None,
        tvec: Optional[List[float]] = None,
        world_points: Optional[np.ndarray] = None,
        image_points: Optional[np.ndarray] = None,
    ):
        # Define distortion coefficients and camera matrix
        self.dist = Distortion(*dist)
        self.mtx = CameraMatrix(*mtx)
        # Initialize the world and image points
        self.world_points, self.image_points = self.init_world_and_image_points(
            world_points, image_points
        )
        # Initialize the rotation and translation vectors
        self.rvec, self.tvec = self.init_rvec_and_tvec(rvec, tvec)

    def init_world_and_image_points(self, world_points, image_points):
        """Initialize world and image points

        Parameters:
        ----------
        world_points : list
            World points [[x1, y1, z1], [x2, y2, z2], ...]
        image_points : list
            Image points [[x1, y1], [x2, y2], ...]

        Returns:
        -------

        """
        # Check if world_points and image_points are provided and have the same length
        if world_points is not None and image_points is not None:
            world_points = np.array(world_points)
            image_points = np.array(image_points)
            if len(world_points) != len(image_points):
                raise ValueError(
                    "World points and image points must have the same length"
                )
            else:
                # Check size of world_points [n, 3]
                if world_points.shape[1] != 3:
                    world_points = world_points.T
                # Check size of image_points [n, 2]
                if image_points.shape[1] != 2:
                    image_points = image_points.T
        return world_points, image_points

    def init_rvec_and_tvec(self, rvec, tvec):
        """Initialize rotation and translation vectors

        Parameters:
        ----------
        rvec : list
            Rotation vector [r11, r12, r13, r21, r22, r23, r31, r32, r33]
        tvec : list
            Translation vector [t1, t2, t3]

        Returns:
        -------
        rvec : RotationMatrix
            Rotation matrix
        tvec : TranslationVector
            Translation vector
        """
        # Check if rvec and tvec are provided
        if rvec is not None and tvec is not None:
            # Define rotation matrix and translation vector
            rvec = RotationMatrix(*rvec)
            tvec = TranslationVector(*tvec)
            return rvec, tvec
        elif self.world_points is not None and self.image_points is not None:
            *_, rvec, tvec = cv.solvePnP(
                self.world_points,
                self.image_points,
                self.mtx.as_array(),
                self.dist.as_array(),
            )
            return RotationMatrix(*rvec), TranslationVector(*tvec)
        else:
            raise ValueError(
                "Rotation and translation vectors or world and image points are required"
            )

    # rvec: Optional[RotationMatrix] = None
    # tvec: Optional[TranslationVector] = None
    #  world_points: Optional[List[WorldPoint]] = None
    # image_points: Optional[List[ImagePoint]] = None

    # @computed_field
    # @cached_property
    # def rvec(self) -> RotationMatrix:
    #     raise NotImplementedError

    # @computed_field
    # @cached_property
    # def tvec(self) -> RotationMatrix:
    #     raise NotImplementedError
    # @computed_field
    # @property
    # def dist_array(self) -> List[float]:
    #     """Distortion coefficients as list"""
    #     return [
    #         self.dist.k1,
    #         self.dist.k2,
    #         self.dist.p1,
    #         self.dist.p2,
    #         self.dist.k3,
    #     ]

    # @computed_field
    # @property
    # def mtx_array(self) -> List[List[float]]:
    #     """Camera matrix as list of lists"""
    #     return [
    #         [self.mtx.fx, 0, self.mtx.cx],
    #         [0, self.mtx.fy, self.mtx.cy],
    #         [0, 0, 1],
    #     ]

    # def estimate_pose(self):
    #     """Estimate the pose of the camera using the provided world coordinates"""
    #     # Check if world_points and image_points are provided
    #     if self.world_points is None or self.image_points is None:
    #         raise ValueError(
    #             "World points and corresponding image points are required to estimate the pose"
    #         )
    #     else:
    #         if len(self.world_points) != len(self.image_points):
    #             raise ValueError(
    #                 "World points and image points must have the same length"
    #             )
    #         else:
    #             if self.rvec is None or self.tvec is None:
    #                 # Estimate the pose
    #                 ret, tvec, rvec = solvePnP(
    #                     self.world_points,
    #                     self.image_points,
    #                     self.mtx_array,
    #                     self.dist_array,
    #                 )
    #                 return ret, tvec, rvec
    #             else:
    #                 raise ValueError(
    #                     "Rotation and translation vectors are already provided"
    #                 )


if __name__ == "__main__":
    # Createa a camera object from a list of parameters
    dist = [-0.0995, 0.2664, 0.0, 0.0, 0.0]
    mtx = [2.7228e3, 2.7237e3, 1.5846e3, 1.0820e3]
    # Create a camera object
    camera = Camera(dist=dist, mtx=mtx)
    test = 0
