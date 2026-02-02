import numpy as np
import cv2

def compute_iou_matrix(boxes):
    '''
    Given a set of bboxes (in [x1, y1, x2, y2]), output an IOU matrix,
    while ignoring pairs where one box fully contains the other.
    '''
    N = boxes.shape[0]
    iou_matrix = np.zeros((N, N), dtype=np.float32)

    def is_contained(box_a, box_b):
        return (
            box_a[0] <= box_b[0] and box_a[1] <= box_b[1] and
            box_a[2] >= box_b[2] and box_a[3] >= box_b[3]
        )

    def is_almost_contained(inner, outer, epsilon=2):
	    x1_i, y1_i, w_i, h_i = inner
	    x1_o, y1_o, w_o, h_o = outer

	    x2_i, y2_i = x1_i + w_i, y1_i + h_i
	    x2_o, y2_o = x1_o + w_o, y1_o + h_o

	    return (
	        x1_i >= x1_o - epsilon and
	        y1_i >= y1_o - epsilon and
	        x2_i <= x2_o + epsilon and
	        y2_i <= y2_o + epsilon
	    )

    for i in range(N):
        x1_i, y1_i, x2_i, y2_i = boxes[i]
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        for j in range(i + 1, N):
            x1_j, y1_j, x2_j, y2_j = boxes[j]
            area_j = (x2_j - x1_j) * (y2_j - y1_j)

            box_i = boxes[i]
            box_j = boxes[j]

            # Skip if one box fully contains the other
            if is_almost_contained(box_i, box_j) or is_almost_contained(box_j, box_i):
                iou = 0.0
            else:
                inter_x1 = max(x1_i, x1_j)
                inter_y1 = max(y1_i, y1_j)
                inter_x2 = min(x2_i, x2_j)
                inter_y2 = min(y2_i, y2_j)

                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                union_area = area_i + area_j - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0

            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou  # symmetric

    return iou_matrix



def draw_bboxes(image, bbox_xyxy, color=(0, 255, 0), thickness=2):
    '''
    given an image and set of bboxes, output a bbox annotated image
    '''
    image_copy = image.copy()
    for box in bbox_xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
    return image_copy


def add_black_border(patch, border_width=1):
    """
    Set the outermost border of the patch to white.
    
    Args:
        patch (np.ndarray): An image patch of shape (H, W, 3) for RGB.
        border_width (int): Width of the white border (default: 1 pixel).
    
    Returns:
        np.ndarray: The patch with a black border.
    """
    patch[:border_width, :, 0:3] = 255              # Top border
    patch[-border_width:, :, 0:3] = 255             # Bottom border
    patch[:, :border_width, 0:3] = 255              # Left border
    patch[:, -border_width:, 0:3] = 255             # Right border
    return patch

def mask_to_bbox_xywh(mask):
    """
    Obtain bbox (xywh) from the mask.

    Args:
        mask (np.ndarray)

    Returns:
        np.ndarray: a bbox of shape (1, 4), in the format of [x, y, w, h]。
                    if empty, return shape (0, 4)。
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 4), dtype=np.int32)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return np.array(bbox, dtype=np.int32)
