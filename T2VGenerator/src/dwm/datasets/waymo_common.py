import numpy as np


def compute_inclination(inclination_range, height, dtype=np.float32):
    diff = inclination_range[1] - inclination_range[0]
    normal_range = (np.arange(0, height, dtype=dtype) + 0.5) / height
    inclination = normal_range * diff + inclination_range[0]
    return inclination


def compute_range_image_polar(
    range_image: np.array, extrinsic: np.array, inclination: np.array,
    dtype=np.float32
):
    _, width = range_image.shape
    az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (np.arange(width, 0, -1, dtype=dtype) - 0.5) / width
    azimuth = (ratios * 2 - 1) * np.pi - az_correction
    return azimuth[None, :], inclination[:, None], range_image


def get_rotation_matrix(
    roll: np.array, pitch: np.array, yaw: np.array,
):
    """Gets a rotation matrix given roll, pitch, yaw.

    roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
    x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

    https://en.wikipedia.org/wiki/Euler_angles
    http://planning.cs.uiuc.edu/node102.html

    Args:
        roll : x-rotation in radians.
        pitch: y-rotation in radians. The shape must be the same as roll.
        yaw: z-rotation in radians. The shape must be the same as roll.

    Returns:
        A rotation tensor with the same data type of the input. Its shape is
        [3 ,3].
    """

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    ones = np.ones_like(cos_yaw)
    zeros = np.zeros_like(cos_yaw)

    r_roll = np.stack([
        np.stack([ones, zeros, zeros], axis=-1),
        np.stack([zeros, cos_roll, -sin_roll], axis=-1),
        np.stack([zeros, sin_roll, cos_roll], axis=-1),
    ], axis=-2)
    r_pitch = np.stack([
        np.stack([cos_pitch, zeros, sin_pitch], axis=-1),
        np.stack([zeros, ones, zeros], axis=-1),
        np.stack([-sin_pitch, zeros, cos_pitch], axis=-1),
    ], axis=-2)
    r_yaw = np.stack([
        np.stack([cos_yaw, -sin_yaw, zeros], axis=-1),
        np.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        np.stack([zeros, zeros, ones], axis=-1),
    ], axis=-2)

    return np.matmul(r_yaw, np.matmul(r_pitch, r_roll))


def compute_range_image_cartesian(
    azimuth: np.array, inclination: np.array, range_image_range: np.array,
    extrinsic: np.array, pixel_pose=None, frame_pose=None,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
        range_image_polar: [B, H, W, 3] float tensor. Lidar range image in
            polar coordinate in sensor frame.
        extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
        pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for
            each range image pixel.
        frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is
            set. It decides the vehicle frame at which the cartesian points are
            computed.
        dtype: float type to use internally. This is needed as extrinsic and
            inclination sometimes have higher resolution than range_image.

    Returns:
        range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """

    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    cos_incl = np.cos(inclination)
    sin_incl = np.sin(inclination)

    # [H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [H, W, 3]
    range_image_points = np.stack([x, y, z], -1)

    # To vehicle frame.
    rotation = extrinsic[0:3, 0:3].T
    translation = extrinsic[0:3, 3:4].T
    range_image_points = range_image_points @ rotation + translation

    if pixel_pose is not None:
        # To global frame.
        # [H, W, 3, 3]
        pixel_pose_rotation = np.swapaxes(
            get_rotation_matrix(
                pixel_pose[..., 0], pixel_pose[..., 1], pixel_pose[..., 2]),
            -1, -2)

        # [H, W, 1, 3]
        pixel_pose_translation = pixel_pose[..., None, 3:]
        range_image_points = (
            range_image_points[..., None, :] @ pixel_pose_rotation +
            pixel_pose_translation)

        # [H, W, 3]
        range_image_points = range_image_points.reshape(
            *range_image_points.shape[:-2], -1)

        if frame_pose is None:
            raise ValueError('frame_pose must be set when pixel_pose is set.')

        # To vehicle frame corresponding to the given frame_pose
        # [4, 4]
        world_to_vehicle = np.linalg.inv(frame_pose)
        world_to_vehicle_rotation = world_to_vehicle[0:3, 0:3].T
        world_to_vehicle_translation = world_to_vehicle[0:3, 3:4].T

        # [H, W, 3]
        range_image_points = range_image_points @ world_to_vehicle_rotation + \
            world_to_vehicle_translation

    return range_image_points


def convert_range_image_to_cartesian(
    range_image: np.array, calibration: dict, lidar_pose=None, frame_pose=None,
    dtype=np.float32
):
    """Converts one range image from polar coordinates to Cartesian coordinates.

    Args:
        range_image: One range image return captured by a LiDAR sensor.
        calibration: Parameters for calibration of a LiDAR sensor.

    Returns:
        A [H, W, 3] image in Cartesian coordinates.
    """

    extrinsic = np.array(
        calibration["[LiDARCalibrationComponent].extrinsic.transform"],
        dtype).reshape(4, 4)

    # Compute inclinations mapping range image rows to circles in the 3D worlds.
    bi_values_key = "[LiDARCalibrationComponent].beam_inclination.values"
    bi_min_key = "[LiDARCalibrationComponent].beam_inclination.min"
    bi_max_key = "[LiDARCalibrationComponent].beam_inclination.max"
    if calibration[bi_values_key] is not None:
        inclination = np.array(calibration[bi_values_key], dtype)
    else:
        inclination = compute_inclination(
            (calibration[bi_min_key], calibration[bi_max_key]),
            range_image.shape[0], dtype)

    inclination = np.flip(inclination, axis=-1)

    # Compute points from the range image
    azimuth, inclination, range_image_range = compute_range_image_polar(
        range_image[..., 0], extrinsic, inclination, dtype)
    range_image_cartesian = compute_range_image_cartesian(
        azimuth, inclination, range_image_range, extrinsic,
        pixel_pose=lidar_pose, frame_pose=frame_pose)
    range_image_mask = range_image[..., 0] > 0

    flatten_range_image_cartesian = range_image_cartesian.reshape(-1, 3)
    flatten_range_image_mask = range_image_mask.reshape(-1)
    return flatten_range_image_cartesian[flatten_range_image_mask]


def laser_calibration_to_dict(lc):
    lc.beam_inclination_max
    return {
        "key.laser_name": lc.name,
        "[LiDARCalibrationComponent].beam_inclination.values":
        lc.beam_inclinations,
        "[LiDARCalibrationComponent].beam_inclination.min":
        lc.beam_inclination_min,
        "[LiDARCalibrationComponent].beam_inclination.max":
        lc.beam_inclination_max,
        "[LiDARCalibrationComponent].extrinsic.transform":
        lc.extrinsic.transform
    }
