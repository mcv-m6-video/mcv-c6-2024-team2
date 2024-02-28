import numpy as np
import xmltodict


def parse_annotations(path, isGT=False):
    # Open the file
    with open(path) as file:
        tracks = xmltodict.parse(file.read())['annotations']['track']

    # Define list of frames and bounding boxes and ground truth
    frames = []
    BBOX = []
    groundTruth = []

    # Loop over the tracks
    for track in tracks:
        id = track['@id']
        label = track['@label']
        boxes = track['box']
        for box in boxes:
            if label == 'car':
                parked = box['attribute']['#text'].lower() == 'true'
            else:
                parked = None
            gt_list = [int(box['@frame']), int(id), label, float(box['@xtl']), float(box['@ytl']),
                           float(box['@xbr']), float(box['@ybr']), float(-1), parked]
            groundTruth.append(gt_list)

    # Loop over the ground truth boxes
    for gt in groundTruth:
        # Extract Frames and bounding boxes
        frame = gt[0]
        bbox = [gt[3], gt[4], gt[5], gt[6]]

        # Add the frames and bounding boxes
        frames.append(frame)
        BBOX.append(bbox)

    # Sort frames
    sortedFrames, sortedBbox = zip(*sorted(zip(frames, BBOX)))

    # Define lists
    bbox = []
    groundTruthComplete = []

    # Help from https://github.com/mcv-m6-video/mcv-m6-2021-team7/tree/main/week1
    # Loop over the sorted Boxes
    for i in range(len(sortedBbox)):
        if i == 0:
            bbox.append(sortedBbox[i])

        else:
            if sortedFrames[i] == sortedFrames[i - 1]:
                bbox.append(sortedBbox[i])
            else:
                if isGT:
                    groundTruthComplete.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(bbox), "already_detected": [False] * len(bbox)})
                else:
                    groundTruthComplete.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox)})
                bbox = [sortedBbox[i]]

            if i + 1 == len(sortedBbox):
                if isGT:
                    groundTruthComplete.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(bbox), "already_detected": [False] * len(bbox)})
                else:
                    groundTruthComplete.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox)})

    return groundTruthComplete, sortedFrames


def getPredictions(path, isGT=False):
    # Open file
    with open(path) as f:
        lines = f.readlines()

    # Define the list of frames, bounding boxes and confidence scores
    frames = []
    BBOX = []
    confidence_scores = []
    predictions = []

    # Loop over the tracks
    for line in lines:
        track = line.split(',')
        pred_list = [int(track[0]) - 1, track[1], 'car', float(track[2]), float(track[3]),
                     float(track[2]) + float(track[4]),
                     float(track[3]) + float(track[5]), float(track[6])]
        predictions.append(pred_list)

    # Loop over the predictions
    for prediction in predictions:
        # Grab the frames, bounding box and confidence for each prediction
        frame = prediction[0]
        bbox = [prediction[3], prediction[4], prediction[5], prediction[6]]
        confidence = prediction[7]

        # Append each variable respectively
        frames.append(frame)
        BBOX.append(bbox)
        confidence_scores.append(confidence)

    # Sort frames and corresponding bounding boxes and confidence scores
    sortedFrames, sortedBbox, sortedScore = zip(*sorted(zip(frames, BBOX, confidence_scores)))

    # Define list of bounding boxes, score
    boundingBox = []
    score = []
    detectedInfo = []

    # Loop over the sorted boxes
    for i in range(len(sortedBbox)):
        # If first frame add bounding box
        if i == 0:
            boundingBox.append(sortedBbox[i])
            score.append(sortedScore[i])

        else:
            if sortedFrames[i] == sortedFrames[i - 1]:
                boundingBox.append(sortedBbox[i])
                score.append(sortedScore[i])
            else:
                if isGT:
                    # Parameter to check if box is already detected for the frame
                    detectedInfo.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(boundingBox), "score": np.array(score),
                         "already_detected": [False] * len(boundingBox)})
                else:
                    detectedInfo.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(boundingBox), "score": np.array(score)})
                boundingBox = []
                score = []
                boundingBox.append(sortedBbox[i])
                score.append(sortedScore[i])

            if i + 1 == len(sortedBbox):
                if isGT:
                    detectedInfo.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(boundingBox), "score": np.array(score),
                         "already_detected": [False] * len(boundingBox)})
                else:
                    detectedInfo.append(
                        {"frame": sortedFrames[i - 1], "bbox": np.array(boundingBox), "score": np.array(score)})

    # Return
    return detectedInfo
