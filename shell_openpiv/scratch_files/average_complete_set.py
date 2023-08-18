"""Determine the average of a complete set of images.

author: Mike van Meerkerk (Deltares)
date: 17-08-2023
"""
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_average_of_n_images(
    image_list: List[str],
    number_of_images_to_average_over: int,
    n_show: int,
    plot_figure: bool = True,
):
    """Compute the average of n images.

    Parameters
    ----------
    image_list : List[str]
        List of images to average over.
    number_of_images_to_average_over : int
        Number of images to average over.

    Returns
    -------
    avg_image : np.ndarray
        Average image in RGB color space.
    """
    # Average n-images in path
    alpha = 1 / number_of_images_to_average_over
    # Initialize average image
    initialized_image = False
    # Loop
    for image_index, image_name in enumerate(
        image_list[:number_of_images_to_average_over]
    ):
        n_avg = image_index + 1
        print(
            f"Processing image {image_index + 1} of {number_of_images_to_average_over}"
        )

        # Src image
        src_image = cv2.cvtColor(cv2.imread(str(image_name)), cv2.COLOR_BGR2RGB)
        # 2D median filter to remove noise while preserving edges
        src_image = cv2.medianBlur(src_image, 3)
        # Initialize average image
        if not initialized_image:
            avg_image = src_image
            initialized_image = True
            # Plot figure
            if plot_figure == True:
                fig1 = plt.figure()
                im_obj = plt.imshow(avg_image)
                ncount = 1
                plt.show(block=False)
                plt.pause(0.001)

        elif image_index < number_of_images_to_average_over:
            avg_image = avg_image * (n_avg - 1) / n_avg + src_image * (1 / n_avg)
            ncount += 1
            if ncount == n_show:
                # plt.close("all")
                # plt.figure()
                # im_obj = plt.imshow(np.uint8(avg_image))
                # plt.show(block=False)
                im_obj.set_data(np.uint8(avg_image))
                fig1.canvas.draw()
                fig1.canvas.flush_events()
                plt.pause(0.001)
                ncount = 1

        else:
            avg_image = (1 - alpha) * avg_image + alpha * src_image
    # Convert image to uint8 and to RGB color space.
    return np.uint8(avg_image)


if __name__ == "__main__":
    # Get images in path
    image_folder = Path(
        r"p:/archivedprojects/11208914-meting-stuw-driel/HDD001/20230127-meting01"
    )
    image_list = list(image_folder.glob("*.tif"))
    # Set number of average images
    n_average = 1_000
    n_show = 50
    # Determine average of n images
    average_image = compute_average_of_n_images(image_list, n_average, n_show)
    # Save average image
    output_path = Path(
        r"n:\Projects\11208500\11208914\C. Report - advise\images\Averaged_Camera_Images"
    )
    cv2.imwrite(
        str(
            Path(
                output_path,
                f"{image_folder.parents[0].name}_{image_folder.name}_average_image_{n_average:05.0f}.tiff",
            )
        ),
        cv2.cvtColor(average_image, cv2.COLOR_RGB2BGR),
    )
