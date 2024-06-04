import PIL
from PIL import Image
import numpy as np
import os

def apply_trimap_to_image(image_path, save_path, output_path):
    image_name = os.path.basename(image_path)
    
    image_filename = "./figs/" + image_name
    trimap_filename = save_path
    output_filename = output_path + image_name
    # Open the RGB image
    image = Image.open(image_filename).convert("RGB")
    
    # Open the trimap
    trimap = Image.open(trimap_filename).convert("L")

    # Convert trimap to binary mask (0 for black and 1 for white)
    trimap_np = np.array(trimap)
    mask = trimap_np > 128  # assuming white is close to 255

    # Convert mask to 3 channels
    mask_3d = np.stack([mask]*3, axis=-1)

    # Convert image to numpy array
    image_np = np.array(image)

    # Apply the mask to the image
    masked_image = image_np * mask_3d

    # Convert the masked image back to PIL format
    masked_image_pil = Image.fromarray(masked_image)

    # Save the masked image
    masked_image_pil.save(output_filename)
    print('Masked image saved: ', output_filename)

