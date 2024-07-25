import numpy as np
from PIL import Image, ImageDraw


def plot_labels(img, labels, class_dict, color=(255, 0, 0), font=None):
    """Plot labels on image

    Args:
        img (np.ndarray): image to plot labels on
        labels (list): label, xmin, ymin, xmax, ymax
        class_dict (dict): class dictionary
        color (tuple, optional): color of labels. Defaults to (255, 0, 0).
        font (PIL.ImageFont, optional): font of labels. Defaults to None.

    Returns:
        np.ndarray: image with labels
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for label in labels:
        draw.rectangle(label[1:], outline=color, width=2)
        draw.text((label[1], label[2] - 10), class_dict[label[0]], fill=(255, 105, 180), font=font)
    return img
