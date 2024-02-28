import xml.etree.ElementTree as ET

def parse_xml_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Sets to hold frames with and without movement
    frames_with_movement = set()
    all_frames = set()
    
    
    frame_with_movement_data = {}
    frame_without_movement_data = {}

    for track in root.findall('track'):
        if track.attrib['label'] in ['car',"bike"]:  # Assuming we're only interested in cars
            for box in track.findall('box'):
                frame = int(box.attrib['frame'])
                all_frames.add(frame)  # Keep track of all frames encountered

                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                try:
                    parked = box.find("attribute[@name='parked']").text == 'true'  # Check if the car is parked

                    if not parked:  # If the car is moving
                        if frame not in frame_with_movement_data:
                            frame_with_movement_data[frame] = []
                        
                        
                        frames_with_movement.add(frame)  # Add the frame to the set of frames with movement
                        frame_with_movement_data[frame].append({
                            "label": track.attrib['label'],
                            "xtl":xtl,
                            "ytl":ytl,
                            "xbr":xbr,
                            "ybr":ybr
                        })
                        
                        
                except: # for bike
                    if frame not in frame_with_movement_data:
                            frame_with_movement_data[frame] = []
                        
                        
                    frames_with_movement.add(frame)  # Add the frame to the set of frames with movement
                    frame_with_movement_data[frame].append({
                        "label": track.attrib['label'],
                        "xtl":xtl,
                        "ytl":ytl,
                        "xbr":xbr,
                        "ybr":ybr
                    })
                    
                
                    
    # Determine frames without movement by subtracting frames with movement from all frames
    frames_without_movement = all_frames - frames_with_movement

    # Convert sets to sorted lists
    frames_with_movement = sorted(list(frames_with_movement))
    frames_without_movement = sorted(list(frames_without_movement))

    return frame_with_movement_data, {frame:[] for frame in frames_without_movement}