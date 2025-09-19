import cv2
import matplotlib.image as mpimg
import numpy as np

import torch_concepts.data.traffic_construction.utils as utils

from torch_concepts.data.traffic_construction.shared import SPRITES_DIRECTORY

################################################################################
## Load the sprites to memory
################################################################################

LIGHTS_FILE = SPRITES_DIRECTORY('lights.png')

################################################################################
## Construct Light Sprites
################################################################################

img = mpimg.imread(LIGHTS_FILE)

RED_LIGHT = mpimg.imread(LIGHTS_FILE)
RED_LIGHT = RED_LIGHT[:, :RED_LIGHT.shape[1]//3, :]

YELLOW_LIGHT = mpimg.imread(LIGHTS_FILE)
YELLOW_LIGHT = \
    YELLOW_LIGHT[:, YELLOW_LIGHT.shape[1]//3:2*YELLOW_LIGHT.shape[1]//3, :]

GREEN_LIGHT = mpimg.imread(LIGHTS_FILE)
GREEN_LIGHT = GREEN_LIGHT[:, 2*GREEN_LIGHT.shape[1]//3:, :]

LIGHTS = [GREEN_LIGHT, YELLOW_LIGHT, RED_LIGHT]
# Convention: 0=green, 1=yellow, 2=red

_LEFT_TOP_CORNER = (400, 400)
_LEFT_BOTTOM_CORNER = (400, 860)
_RIGHT_TOP_CORNER = (875, 400)
_RIGHT_BOTTOM_CORNER = (875, 860)

################################################################################
## Helper functions
################################################################################


def add_circle_to_image(
    background,
    center,
    radius,
    color,
    vector_angle=None,
    arrow_color=(0, 0, 0),
    thickness=10,
    inplace=False,
    tip_length=0.4,
):
    if inplace:
        image = background
    else:
        image = background.copy()

    # Draw the circle
    cv2.circle(
        image[:, :, :3],
        center,
        radius,
        color,
        -1,  # -1 means filled circle
    )

    if vector_angle is not None:
        angle_rad = np.radians(vector_angle)
        start_x = int(center[0] - radius * np.cos(angle_rad))
        start_y = int(center[1] + radius * np.sin(angle_rad))
        end_x = int(center[0] + radius * np.cos(angle_rad))
        end_y = int(center[1] - radius * np.sin(angle_rad))

        cv2.arrowedLine(
            image[:, :, :3],
            (start_x, start_y),
            (end_x, end_y),
            arrow_color,
            thickness,
            tipLength=tip_length,
        )
    return image


################################################################################
## Functions to export
################################################################################



def add_light_x_axis(
    img,
    green,
    ratio=1,
    light_scale=1,
    inplace=False,
    circle_radius=50,
    use_lights_sprites=False,
    thickness=10,
):
    # green: array of 4 indices, each 0=green, 1=yellow, 2=red, for each direction
    # Convention: [car_of_interest, right, opposite, left]
    # For x axis: left=west, right=east
    # Place two lights: left (west) and right (east)
    # left (west) = index 0, right (east) = index 2
    if use_lights_sprites:
        # West (left, car of interest)
        light = LIGHTS[green[0]]
        x_offset, y_offset = 330, 850
        x_scale_shift, y_scale_shift = 0, 0
        if light_scale != 1:
            x_scale_shift = -int(0.1 * light.shape[0] * (light_scale - 1))
            y_scale_shift = int(0.1 * light.shape[1] * (light_scale - 1))
        x_offset += x_scale_shift
        y_offset += y_scale_shift
        if ratio != 1:
            x_offset, y_offset = utils.transform_scale_coordinates(
                x_offset,
                y_offset,
                ratio=ratio,
            )
            light = utils.resize_with_aspect_ratio(
                light,
                target_height=ratio,
            )
        img = utils.add_sprite(
            sprite=light,
            background=img,
            target_width=0.1 * light_scale,
            rotation=270,
            x_offset=x_offset,
            y_offset=y_offset,
            inplace=inplace,
        )
        # East (opposite)
        light = LIGHTS[green[2]]
        x_offset, y_offset = 840, 360
        x_scale_shift, y_scale_shift = 0, 0
        if light_scale != 1:
            x_scale_shift = int(0.1 * light.shape[0] * (light_scale - 1))
            y_scale_shift = -int(0.1 * light.shape[1] * (light_scale - 1))
        x_offset += x_scale_shift
        y_offset += y_scale_shift
        if ratio != 1:
            x_offset, y_offset = utils.transform_scale_coordinates(
                x_offset,
                y_offset,
                ratio=ratio,
            )
        img = utils.add_sprite(
            sprite=light,
            background=img,
            target_width=0.1 * light_scale,
            rotation=90,
            x_offset=x_offset,
            y_offset=y_offset,
            inplace=inplace,
        )
        return img
    # Circles
    r = int(circle_radius * ratio * light_scale)
    # West (left, car of interest)
    color = [(0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)][green[0]]
    x_offset, y_offset = _LEFT_BOTTOM_CORNER
    if ratio != 1:
        x_offset, y_offset = utils.transform_scale_coordinates(
            x_offset,
            y_offset,
            ratio=ratio,
        )
    img = add_circle_to_image(
        background=img,
        center=(x_offset - r, y_offset + r),
        radius=r,
        color=color,
        inplace=inplace,
        vector_angle=0,
        thickness=max(1, int(thickness * ratio)),
    )
    # East (opposite)
    color = [(0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)][green[2]]
    x_offset, y_offset = _RIGHT_TOP_CORNER
    if ratio != 1:
        x_offset, y_offset = utils.transform_scale_coordinates(
            x_offset,
            y_offset,
            ratio=ratio,
        )
    img = add_circle_to_image(
        background=img,
        center=(x_offset + r, y_offset - r),
        radius=r,
        color=color,
        inplace=inplace,
        vector_angle=180,
        thickness=max(1, int(thickness * ratio)),
    )
    return img



def add_light_y_axis(
    img,
    green,
    ratio=1,
    light_scale=1,
    inplace=False,
    circle_radius=50,
    use_lights_sprites=False,
    thickness=10,
):
    # green: array of 4 indices, each 0=green, 1=yellow, 2=red, for each direction
    # Convention: [car_of_interest, right, opposite, left]
    # For y axis: top=north, bottom=south
    # Place two lights: top (north, index 1), bottom (south, index 3)
    if use_lights_sprites:
        # North (right)
        light = LIGHTS[green[1]]
        x_offset, y_offset = 850, 850
        x_scale_shift, y_scale_shift = 0, 0
        if light_scale != 1:
            x_scale_shift = -int(0.1 * light.shape[0] * (light_scale - 1))
            y_scale_shift = -int(0.1 * light.shape[1] * (light_scale - 1))
        x_offset += x_scale_shift
        y_offset += y_scale_shift
        if ratio != 1:
            x_offset, y_offset = utils.transform_scale_coordinates(
                x_offset,
                y_offset,
                ratio=ratio,
            )
            light = utils.resize_with_aspect_ratio(
                light,
                target_height=ratio,
            )
        img = utils.add_sprite(
            sprite=light,
            background=img,
            target_width=0.1*light_scale,
            rotation=0,
            x_offset=x_offset,
            y_offset=y_offset,
            inplace=inplace,
        )
        # South (left)
        light = LIGHTS[green[3]]
        x_offset, y_offset = 370, 300
        if ratio != 1:
            x_offset, y_offset = utils.transform_scale_coordinates(
                x_offset,
                y_offset,
                ratio=ratio,
            )
        x_scale_shift, y_scale_shift = 0, 0
        if light_scale != 1:
            x_scale_shift = int(0.1 * light.shape[0] * (light_scale - 1))
            y_scale_shift = -int(0.1 * light.shape[1] * (light_scale - 1))
        x_offset += x_scale_shift
        y_offset += y_scale_shift
        img = utils.add_sprite(
            sprite=light,
            background=img,
            target_width=0.1*light_scale,
            rotation=180,
            x_offset=x_offset,
            y_offset=y_offset,
            inplace=inplace,
        )
        return img
    # Circles
    r = int(circle_radius * ratio * light_scale)
    # North (right)
    color = [(0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)][green[1]]
    x_offset, y_offset = _LEFT_TOP_CORNER
    if ratio != 1:
        x_offset, y_offset = utils.transform_scale_coordinates(
            x_offset,
            y_offset,
            ratio=ratio,
        )
    img = add_circle_to_image(
        background=img,
        center=(x_offset - r, y_offset - r),
        radius=r,
        color=color,
        inplace=inplace,
        vector_angle=270,
        thickness=max(1, int(thickness * ratio)),
    )
    # South (left)
    color = [(0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)][green[3]]
    x_offset, y_offset = _RIGHT_BOTTOM_CORNER
    if ratio != 1:
        x_offset, y_offset = utils.transform_scale_coordinates(
            x_offset,
            y_offset,
            ratio=ratio,
        )
    img = add_circle_to_image(
        background=img,
        center=(x_offset + r, y_offset + r),
        radius=r,
        color=color,
        inplace=inplace,
        vector_angle=90,
        thickness=max(1, int(thickness * ratio)),
    )
    return img
