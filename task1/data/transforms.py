import colorsys
from PIL import Image, ImageChops, ImageOps
import numpy as np
import cv2

def apply_grayscale(img: Image.Image) -> Image.Image:
    # Convert to grayscale and back to RGB to replicate channels
    return img.convert('L').convert('RGB')

def apply_hue_rotation(img: Image.Image, degrees=120) -> Image.Image:
    # Convert to HSV using cv2
    img_np = np.array(img.convert('RGB'))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Hue is [0, 180] in OpenCV
    hsv[:, :, 0] = (hsv[:, :, 0] + (degrees / 2)) % 180
    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb)

def apply_translation(img: Image.Image, delta: int, direction: str) -> Image.Image:
    w, h = img.size
    img_np = np.array(img)
    
    pad_w = delta
    pad_h = delta
    
    # reflection padding
    padded = np.pad(img_np, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    
    # Crop at offset
    start_y = pad_h
    start_x = pad_w
    if direction == 'up':
        start_y -= delta
    elif direction == 'down':
        start_y += delta
    elif direction == 'left':
        start_x -= delta
    elif direction == 'right':
        start_x += delta
        
    cropped = padded[start_y:start_y+h, start_x:start_x+w]
    return Image.fromarray(cropped)

def apply_patch_shuffle(img: Image.Image, permutation: list) -> Image.Image:
    w, h = img.size
    patch_w = w // 4
    patch_h = h // 4
    
    patches = []
    for i in range(4):
        for j in range(4):
            box = (j*patch_w, i*patch_h, (j+1)*patch_w, (i+1)*patch_h)
            patches.append(img.crop(box))
            
    shuffled_img = Image.new('RGB', (w, h))
    for idx, patch_idx in enumerate(permutation):
        i = idx // 4
        j = idx % 4
        shuffled_img.paste(patches[patch_idx], (j*patch_w, i*patch_h))
        
    return shuffled_img
