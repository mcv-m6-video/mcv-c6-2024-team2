import json
from tqdm import tqdm


def track_to_file(tracked, out):
    with open(out, "w") as f:
        for frame_id in tqdm(tracked.keys()):
            for bbox in tracked[frame_id]:
                obj_id = int(bbox[4])
                bbox = list(map(int, bbox[:4]))

                f.write(f'{int(frame_id)}, {obj_id}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, 1, -1, -1, -1\n')

    print("Done with saving")


def get_bboxes(path, with_id=False):
    with open(path, 'r') as file:
        data = file.read()

    # Splitting the data into lines
    lines = data.split('\n')[:-1] # Until -1 to remove the last empty line

    # Parsing the values
    bboxes = {}
    for line in lines:
        parts = line.split(',')

        frame = int(parts[0])
        id = int(parts[1])
        left = float(parts[2])
        top = float(parts[3])
        width = float(parts[4])
        height = float(parts[5])

        if frame not in bboxes:
            bboxes[frame] = []

        if with_id:
            bboxes[frame].append([id, left, top, width, height])
        else:
            bboxes[frame].append([left, top, width, height])

    return bboxes