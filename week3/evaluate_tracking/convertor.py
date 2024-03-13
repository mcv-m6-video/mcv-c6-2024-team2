import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path


def track_to_csv(file, out):
    with open(file) as f:
        data = json.load(f)

    with open(out, "w") as f:
        for frame in tqdm(data):
            written_ids = []
            for track in data[frame]:
                id = track['tracked_id']
                if id in written_ids:
                    continue
                written_ids.append(id)
                
                f.write(f'{int(frame) + 1}, {id}, {track["xtl"]}, {track["ytl"]}, {track["xbr"] - track["xtl"]}, {track["ybr"] - track["ytl"]}, 1, -1, -1, -1\n')


# Taken from Team1
def XML_to_csv(annots, out, remParked=False):
    file = ET.parse(annots)
    root = file.getroot()

    with open(out, "w") as f:
        for child in tqdm(root):
            if child.tag == "track":
                # Get class
                id = int(child.attrib["id"])
                className = child.attrib["label"]
                for obj in child:
                    if className == "car":
                        objParked = obj[0].text
                        # Do not store if it is parked and we want to remove parked objects
                        if objParked == "true" and remParked:
                            continue
                    
                    frame = obj.attrib["frame"]
                    xtl = float(obj.attrib["xtl"])
                    ytl = float(obj.attrib["ytl"])
                    xbr = float(obj.attrib["xbr"])
                    ybr = float(obj.attrib["ybr"])
                    f.write(f"{int(frame) + 1}, {id}, {xtl}, {ytl}, {xbr - xtl}, {ybr - ytl}, 1, -1, -1, -1\n")


