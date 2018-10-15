import openslide
from openslide import OpenSlideError
import json
import boto3
import botocore
from openslide.deepzoom import DeepZoomGenerator
import math
import os
import numpy as np


def open_slide(slide_name, folder):
    """
    :param slide_name: Name of slide e.g. 80101_TCGA-63-5131-01Z-00-DX1.C1C3724A-D9FC-46D6-9D7B-9357A58ACAEF.svs
    :param folder: folder location : e.g. /Users/swapnil/CMU_CourseWork/Fall2018/Capstone/PathAI/GeneGAN/tcga_data/
    :return: An OpenSlide object representing a whole-slide image.
    """
    filename = os.path.join(folder,slide_name)
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide


def read_json_file(file_name, folder=None):
    '''

    :param file_name: Test file name like lusc_slide or annotation
    :param folder: folder_name
    :return: returns the text of the file as list
    '''
    filename = os.path.join(folder,file_name)
    with open(filename, 'r') as an:
        file = json.load(an)
    return file


def get_annotated_slides(annotations):
    """

    :param annotations: file with annotations
    :return: returns a set of all slide ids with annotations
    """
    annotated_slides = set([slides['slideId'] for slides in annotations])
    return annotated_slides


def get_tcga_slide_names(lusc_slides, annotations=None):
    """

    :param lusc_slides:
    :param annotations:
    :return: Returns the complete slide name from the 2 datasets e.g 80101_TCGA-63-5131-01Z-00-DX1.C1C3724A-D9FC-46D6-9D7B-9357A58ACAEF.svs
    """
    download_slide_names = []
    for slide in lusc_slides:
        if annotations:
            if slide['slideId'] in annotations:
                download_slide_names.append(str(slide['slideId']) + '_' + slide['originalName'])
        else:
            download_slide_names.append(str(slide['slideId']) + '_' + slide['originalName'])

    return download_slide_names


def create_tile_generator(slide, tile_size=256, overlap=1):
    """
    Create a tile generator for the given slide.
    This generator is able to extract tiles from the overall
    whole-slide image.
    Args:
    slide: An OpenSlide object representing a whole-slide image.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.
    Returns:
    A DeepZoomGenerator object representing the tile generator. Each
    extracted tile is a PIL Image with shape
    (tile_size, tile_size, channels).
    Note: This generator is not a true "Python generator function", but
    rather is an object that is capable of extracting individual tiles.
    """
    generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
    return generator


def get_20x_zoom_level(slide, generator):
    """
    Return the zoom level that corresponds to a 20x magnification.
    The generator can extract tiles from multiple zoom levels,
    downsampling by a factor of 2 per level from highest to lowest
    resolution.
    Args:
    slide: An OpenSlide object representing a whole-slide image.
    generator: A DeepZoomGenerator object representing a tile generator.
      Note: This generator is not a true "Python generator function",
      but rather is an object that is capable of extracting individual
      tiles.
    Returns:
    Zoom level corresponding to a 20x magnification, or as close as
    possible.
    """
    highest_zoom_level = generator.level_count - 1  # 0-based indexing
    try:
        mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        # `mag / 20` gives the downsampling factor between the slide's
        # magnification and the desired 20x magnification.
        # `(mag / 20) / 2` gives the zoom level offset from the highest
        # resolution level, based on a 2x downsampling factor in the
        # generator.
        offset = math.floor((mag / 20) / 2)
        level = highest_zoom_level - offset
    except (ValueError, KeyError) as e:
        # In case the slide magnification level is unknown, just
        # use the highest resolution.
        level = highest_zoom_level
    return level


def generate_slide_indices(slide_name, folder, tile_size=256, overlap=1):
    """
    Generate all possible tile indices for a whole-slide image.
    Given a slide number, tile size, and overlap, generate
    all possible (slide_num, tile_size, overlap, zoom_level, col, row)
    indices.
    Args:
    slide_num: Slide image number as an integer.
    folder: Directory in which the slides folder is stored, as a string.
      This should contain either a `training_image_data` folder with
      images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
      folder with images in the format `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.
    Returns:
    A list of (slide_num, tile_size, overlap, zoom_level, col, row)
    integer index tuples representing possible tiles to extract.
    """
    # Open slide.
    slide = open_slide(slide_name, folder)
    # Create tile generator.
    generator = create_tile_generator(slide, tile_size, overlap)
    # Get 20x zoom level.
    zoom_level = get_20x_zoom_level(slide, generator)
    # Generate all possible (zoom_level, col, row) tile index tuples.
    cols, rows = generator.level_tiles[zoom_level]
    tile_indices = [(slide_name, tile_size, overlap, zoom_level, col, row)
                    for col in range(cols) for row in range(rows)]
    return tile_indices

gene
