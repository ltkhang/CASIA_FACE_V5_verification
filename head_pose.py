import cv2
import math
import numpy as np


def detect_head_pose(size, bbox, points):
    """
    :param size: img.shape
    :param bbox: [x1, y1, x2, y2]
    :param points: landmark reshape((2, 5)).T
    :return: pitch, yaw, roll: float
    """
    image_points = []
    image_points.append((points[2][0], points[2][1]))  # nose
    image_points.append(((points[3][0] + points[4][0]) / 2, bbox[3]))  # chin
    image_points.append((points[0][0], points[0][1]))  # left eye
    image_points.append((points[1][0], points[1][1]))  # right eye
    image_points.append((points[3][0], points[3][1]))  # left mouth
    image_points.append((points[4][0], points[4][1]))  # right mouth

    image_points = np.asarray(image_points, dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return pitch, yaw, roll
