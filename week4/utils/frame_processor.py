import cv2

def height_based_resize(img, new_height):
    # Calculate aspect ratio
    height, width, _ = img.shape
    aspect_ratio = width / height
    
    # Calculate new width based on the desired height
    new_width = int(new_height * aspect_ratio)
    
    # Resize the image while maintaining aspect ratio
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_img