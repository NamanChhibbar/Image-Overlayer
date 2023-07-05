import numpy as np
from PIL import Image, ImageFont, ImageDraw

from .utils import process_text, create_mask, calc_pos, process_image, best_spot

def overlay_text(image: np.ndarray, text: str, font_face: str, text_color: tuple|list=(255, 255, 255), size_choice: str="large", mask_choice: str="none", mask_frac: float=0.5):
    """
    Overlays text on an image with masking options none, whole, top, bottom, left, right.

    Parameters
        image: Numpy array respresentation of the image in RGB format.
        text: Text to overlay.
        font_face: Path to ttf or otf file.
        text_color: Color of text to overlay in the form (R, G, B).
        size_choice: Size of text to overlay. Must be one of small, medium, or large.
        mask_choice: Choice of mask to apply. Must be one of none, whole, top, bottom, left, or right.
        mask_frac: Fraction of dimension to be covered in cases of top, bottom, left, or right masking options.

    Returns
        output: PIL image with text overlayed
    """

    # Limits
    match size_choice:
        case "small":
            word_lim = 30
        case "medium":
            word_lim = 25
        case "large":
            word_lim = 20
    words_per_line = 5
    chars_per_line = 18

    # Safety checks
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array in RGB format")
    if not 0 < len(text.split()) < word_lim:
        raise ValueError(f"Number of words must be between 1 and {word_lim}")
    if mask_choice not in ("none", "whole", "top", "bottom", "left", "right"):
        raise ValueError(f"mask_choice must be one of none, whole, top, bottom, left, or right")
    if not 0 < mask_frac <= 1:
        raise ValueError("mask_frac must be between 0 and 1")
    if size_choice not in ("small", "medium", "large"):
        raise ValueError("size_choice must be one of small, medium, or large")

    # Calculating other parameters
    mask_frac = 1 if mask_choice in ("none", "whole") else mask_frac
    im_dim = image.shape[1::-1]
    mask_dim = np.array(im_dim)
    ind = 0 if mask_choice in ("left", "right") else 1
    mask_dim[ind] = int(mask_frac * mask_dim[ind])
    padding = int(3e-2 * mask_dim[1])
    brightness = 1 - np.log(np.sum(image) / image.size) / 10
    lines, font_size = process_text(text, size_choice, mask_dim, words_per_line, chars_per_line)
    text_font = ImageFont.truetype(font_face, font_size)

    # Creating mask
    if mask_choice != "none":
        image = create_mask(image, im_dim, mask_choice, mask_dim, brightness)

    # Output image
    output = Image.fromarray(image)

    # Writer object to write on image
    writer = ImageDraw.Draw(output)

    # Calculating dimensions of rectangle and position of lines relative to rectangle
    pos, rec_dim = calc_pos(lines, padding, writer, text_font)

    # Calculating position of rectangle
    rec_pos = ((mask_dim - rec_dim) // 2).astype(int)
    match mask_choice:
        case "right":
            rec_pos[0] += im_dim[0] - mask_dim[0]
        case "bottom":
            rec_pos[1] += im_dim[1] - mask_dim[1]

    # Overlaying lines on image
    pos += rec_pos
    for i, p in enumerate(pos):
        writer.text(p, lines[i], fill=text_color, font=text_font)

    return output

def overlay_text_optimally(image: np.ndarray, text: str, font_face: str, text_color: tuple|list=(255, 255, 255), size_choice: str="large"):
    """
    Overlays text on an image on the most optimal spot (with least objects)

    Parameters
        image: Numpy array respresentation of the image in RGB format
        text: Text to overlay
        font_face: Path to ttf or otf file
        text_color: Color of text to overlay in the form (R, G, B)
        size_choice: Size of text to overlay. Must be one of small, medium, or large

    Returns
        output: PIL image with text overlayed
    """

    # Limits
    match size_choice:
        case "small":
            word_lim = 30
        case "medium":
            word_lim = 25
        case "large":
            word_lim = 20
    words_per_line = 5
    chars_per_line = 18

    # Safety checks
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array in RGB format")
    if not 0 < len(text.split()) < word_lim:
        raise ValueError(f"Number of words must be between 1 and {word_lim}")
    if size_choice not in ("small", "medium", "large"):
        raise ValueError("size_choice must be one of small, medium, or large")

    # Parameters for edges finding algorithm
    strip_frac = 0.2
    blur_frac = 0.4
    blur_ints = 6
    n = 150
    num_clst = 5
    seed = 12345

    # Calculating other parameters
    im_dim = np.array(image.shape[1::-1])
    padding = int(3e-2 * im_dim[1])
    brightness = 1 - np.log(np.sum(image) / image.size) / 10
    lines, font_size = process_text(text, size_choice, im_dim, words_per_line, chars_per_line)
    text_font = ImageFont.truetype(font_face, font_size)

    # Dimming image
    image = create_mask(image, im_dim, "whole", 1, brightness)

    # Output image
    output = Image.fromarray(image)

    # Writer object to write on image
    writer = ImageDraw.Draw(output)

    # Calculating dimensions of rectangle and position of lines relative to rectangle
    pos, rec_dim = calc_pos(lines, padding, writer, text_font)

    # Calculating position of rectangle using edges finding algorithm
    image, strip_l, strip_b = process_image(image, strip_frac, blur_frac, blur_ints)
    rec_pos = best_spot(image, n, rec_dim, num_clst, seed) + [strip_l, strip_b]

    # Overlaying lines on image
    pos += rec_pos
    for i, p in enumerate(pos):
        writer.text(p, lines[i], fill=text_color, font=text_font)

    return output
