import xml.etree.ElementTree as ET

def parse_xml_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
        
    boxes = {}

    def process_box(box, label):
        xtl = float(box.attrib['xtl'])
        ytl = float(box.attrib['ytl'])
        xbr = float(box.attrib['xbr'])
        ybr = float(box.attrib['ybr'])

        return {
            "label": label,
            "xtl":xtl,
            "ytl":ytl,
            "xbr":xbr,
            "ybr":ybr
        }

    for track in root.findall('track'):
        # if track.attrib['label'] == 'car':
        for box in track.findall('box'):
            # Check if the car is parked
            # parked = box.find("attribute[@name='parked']")  

            # # If the car is moving or it's a bike
            # if track.attrib['label'] == 'bike' or parked.text != 'true': 
            frame = int(box.attrib['frame'])         
            
            if frame not in boxes:
                boxes[frame] = []

            box_processed = process_box(box, track.attrib['label'])
            boxes[frame].append(box_processed)
            
    return boxes
                    
                
            