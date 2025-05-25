# Datasets

The datasets of this project are used to read multi-view video clips, point cloud sequence clips, as well as corresponding text descriptions for each frame, camera parameters, ego-vehicle pose transform, projected 3D box condition, and projected HD map condition.
We have adapted the nuScenes, Waymo Percepton, and Argoverse 2 Sensor datasets.

## Terms

**Sequence length** is the frame count of the temporal sequence, denoted as `t`.

**View count** is the camera view count, denoted as `v`.

**Sensor count** is the sum of camera view count and the LiDAR count. The LiDAR count can only be 0 or 1 in this version.

**Image coordinate system** is x-right, y-down, with the origin at the top left corner of the image.

**Camera coordinate system** is x-right, y-down, z-forward, with the origin at the camera's optical center.

**Ego vehicle coordinate system** is x-forward, y-left, z-up, with the origin at the the midpoint of the rear vehicle axle.

## Data items

### Basic information

`fps`. A float32 tensor in the shape `(1,)`.

`pts`. A float32 tensor in the shape `(t, sensor_count)`

`images`. A multi-level list of PIL images in the shape `(t, v)`, with the original image size.

`lidar_points`. A list in the shape `(t,)` of float32 tensors in the shape `(point_count, 3)`. The point count in each frame of a sequence is not fixed.

### Transforms

`camera_transforms`. A float32 tensor in the shape `(t, v, 4, 4)`, representing the 4x4 transformation matrix from the camera coordinate system to the ego vehicle coordinate system.

`camera_intrinsics`. A float32 tensor in the shape `(t, v, 3, 3)`, representing the 3x3 transformation matrix from the camera coordinate system to the image coordinate system.

`image_size`. A float32 tensor in the shape `(t, v, 2)` for the pixel width and height of each image.

`lidar_transforms`. A float32 tensor in the shape `(t, 1, 4, 4)`, representing the 4x4 transformation matrix from the LiDAR coordinate system to the ego vehicle coordinate system.

`ego_transforms`. A float32 tensor in the shape `(t, sensor_count, 4, 4)`, representing the 4x4 transformation matrix from the ego vehicle coordinate system to the world coordinate system.

### Conditions

`3dbox_images`. A multi-level list of PIL images in the shape `(t, v)`, the same as the original image size, contains the projected 3D box of object annotations.

`hdmap_images`. A multi-level list of PIL images in the shape `(t, v)`, the same as the original image size, contains the projected HD map of line annotations.

`image_description`. A multi-level list of strings in the shape `(t, v)`, for the text descriptions per image.
