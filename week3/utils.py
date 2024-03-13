def calculate_idf1(gt_bboxes, tracked_bboxes, threshold=0.5):
    """
    Calculate IDF1 (Intersection over Detection F1) score.

    Parameters:
        gt_bboxes (list): List of ground truth bounding boxes in the format [x1, y1, x2, y2].
        tracked_bboxes (list): List of tracked bounding boxes in the format [x1, y1, x2, y2].
        threshold (float): IoU threshold for considering a detection as a true positive.

    Returns:
        float: IDF1 score.
    """
    num_gt = len(gt_bboxes)
    num_tracked = len(tracked_bboxes)

    if num_gt == 0 or num_tracked == 0:
        return 0.0

    # Initialize variables for true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0

    # Iterate through each tracked bounding box
    for tracked_bbox in tracked_bboxes:
        # Compute intersection over union (IoU) with all ground truth bounding boxes
        max_iou = 0
        for gt_bbox in gt_bboxes:
            iou = calculate_iou(tracked_bbox, gt_bbox)
            max_iou = max(max_iou, iou)

        if max_iou >= threshold:
            tp += 1
        else:
            fp += 1

    # Compute false negatives (ground truth not matched with any tracked bounding box)
    fn = num_gt - tp

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        bbox1 (list): Bounding box in the format [x1, y1, x2, y2].
        bbox2 (list): Bounding box in the format [x1, y1, x2, y2].

    Returns:
        float: Intersection over Union (IoU) score.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area_bbox1 + area_bbox2 - intersection

    iou = intersection / union if union > 0 else 0

    return iou

def calculate_idtp(tracked_bboxes, gt_bboxes, threshold=0.5):
    """
    Calculate Identified True Positives (IDTP).

    Parameters:
        tracked_bboxes (list): List of tracked bounding boxes.
        gt_bboxes (list): List of ground truth bounding boxes.
        threshold (float): IoU threshold for considering a detection as a true positive.

    Returns:
        int: Number of Identified True Positives (IDTP).
    """
    idtp = 0
    for tracked_bbox in tracked_bboxes:
        for gt_bbox in gt_bboxes:
            iou = calculate_iou(tracked_bbox, gt_bbox)
            if iou >= threshold:
                idtp += 1
                break  # Matched with a ground truth box, move to the next tracked bbox
    return idtp



def calculate_idfp(tracked_bboxes, gt_bboxes, threshold=0.5):
    """
    Calculate Identified False Positives (IDFP).

    Parameters:
        tracked_bboxes (list): List of tracked bounding boxes.
        gt_bboxes (list): List of ground truth bounding boxes.
        threshold (float): IoU threshold for considering a detection as a true positive.

    Returns:
        int: Number of Identified False Positives (IDFP).
    """
    idfp = len(tracked_bboxes)
    for tracked_bbox in tracked_bboxes:
        for gt_bbox in gt_bboxes:
            iou = calculate_iou(tracked_bbox, gt_bbox)
            if iou >= threshold:
                idfp -= 1  # Remove from IDFP count
                break  # Matched with a ground truth box, move to the next tracked bbox
    return idfp

def calculate_idfn(tracked_bboxes, gt_bboxes, threshold=0.5):
    """
    Calculate Identified False Negatives (IDFN).

    Parameters:
        tracked_bboxes (list): List of tracked bounding boxes.
        gt_bboxes (list): List of ground truth bounding boxes.
        threshold (float): IoU threshold for considering a detection as a true positive.

    Returns:
        int: Number of Identified False Negatives (IDFN).
    """
    idfn = len(gt_bboxes)
    for gt_bbox in gt_bboxes:
        for tracked_bbox in tracked_bboxes:
            iou = calculate_iou(tracked_bbox, gt_bbox)
            if iou >= threshold:
                idfn -= 1  # Remove from IDFN count
                break  # Matched with a tracked bbox, move to the next ground truth bbox
    return idfn


def calculate_hota(tracked_bboxes, gt_bboxes):
    idtp = calculate_idtp(tracked_bboxes, gt_bboxes)
    idfp = calculate_idfp(tracked_bboxes, gt_bboxes)
    idfn = calculate_idfn(tracked_bboxes, gt_bboxes)

    # Compute HOTA score for the current frame
    try:
        hota = idtp / (idtp + 0.5 * (idfp + idfn))
    except ZeroDivisionError:
        hota = 0.0
    return hota