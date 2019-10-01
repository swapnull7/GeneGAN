import openslide
from openslide import OpenSlideError
import json
import boto3
import botocore
from openslide.deepzoom import DeepZoomGenerator
import math
import os
import numpy as np
import tempfile
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk


def optical_density(tile):
    """
    Convert a tile to optical density values.
    Args:
    tile: A 3D NumPy array of shape (tile_size, tile_size, channels).
    Returns:
    A 3D NumPy array of shape (tile_size, tile_size, channels)
    representing optical density values.
    """
    tile = tile.astype(np.float64)
    od = -np.log((tile+1)/240)
    return od


def open_slide(slide_name, folder):
    """
    :param slide_name: Name of slide e.g. 80101_TCGA-63-5131-01Z-00-DX1.C1C3724A-D9FC-46D6-9D7B-9357A58ACAEF.svs
    :param folder: folder location : e.g. /Users/swapnil/CMU_CourseWork/Fall2018/Capstone/PathAI/GeneGAN/tcga_data/
    :return: An OpenSlide object representing a whole-slide image.
    """
    filename = os.path.join(folder, slide_name)
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


def keep_tile(tile_tuple, tile_size, tissue_threshold):
    """
    Determine if a tile should be kept.
    This filters out tiles based on size and a tissue percentage
    threshold, using a custom algorithm. If a tile has height &
    width equal to (tile_size, tile_size), and contains greater
    than or equal to the given percentage, then it will be kept;
    otherwise it will be filtered out.
    Args:
    tile_tuple: A (slide_num, tile) tuple, where slide_num is an
      integer, and tile is a 3D NumPy array of shape
      (tile_size, tile_size, channels).
    tile_size: The width and height of a square tile to be generated.
    tissue_threshold: Tissue percentage threshold.
    Returns:
    A Boolean indicating whether or not a tile should be kept for
    future usage.
    """
    slide_num, tile = tile_tuple
    if tile.shape[0:2] == (tile_size, tile_size):
        tile_orig = tile

    # Check 1
    # Convert 3D RGB image to 2D grayscale image, from
    # 0 (dense tissue) to 1 (plain background).
        tile = rgb2gray(tile)
    # 8-bit depth complement, from 1 (dense tissue)
    # to 0 (plain background).
        tile = 1 - tile
    # Canny edge detection with hysteresis thresholding.
    # This returns a binary map of edges, with 1 equal to
    # an edge. The idea is that tissue would be full of
    # edges, while background would not.
        tile = canny(tile)
    # Binary closing, which is a dilation followed by
    # an erosion. This removes small dark spots, which
    # helps remove noise in the background.
        tile = binary_closing(tile, disk(10))
    # Binary dilation, which enlarges bright areas,
    # and shrinks dark areas. This helps fill in holes
    # within regions of tissue.
        tile = binary_dilation(tile, disk(10))
    # Fill remaining holes within regions of tissue.
        tile = binary_fill_holes(tile)
    # Calculate percentage of tissue coverage.
        percentage = tile.mean()
        check1 = percentage >= tissue_threshold

    # Check 2
    # Convert to optical density values
        tile = optical_density(tile_orig)
    # Threshold at beta
        beta = 0.15
        tile = np.min(tile, axis=2) >= beta
    # Apply morphology for same reasons as above.
        tile = binary_closing(tile, disk(2))
        tile = binary_dilation(tile, disk(2))
        tile = binary_fill_holes(tile)
        percentage = tile.mean()
        check2 = percentage >= tissue_threshold

        return check1 and check2
    else:
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Arguments for generating patches from slides')
    parser.add_argument(
        '-s',
        '--raw_slides_folder',
        type=str,
        default='',
        help='folder containing the WSIs')
    parser.add_argument(
        '-p',
        '--patches_folder',
        type=str,
        default='',
        help='folder containing patches generated from slides')
    parser.add_argument(
        '-t',
        '--slides_tracker_file',
        type=str,
        default='./slides_new.list',
        help='File containing all names of slides sampled from')
    parser.add_argument(
        '-m',
        '--mask_folder',
        type=str,
        default='',
        help='folder containing patches generated from slides')

    args = parser.parse_args()
    return args


def write_parsed_slides(slideID, parsed_slides_tracker_file):
    """

    :param slideID: String with slife
    :return:
    """
    with open(parsed_slides_tracker_file, 'a') as f:
        f.write(str(slideID)+"\n")
    return


def find_parsed_slides(tracker_file):
    """

    :param tracker_file: File to keep track of parsed slides incase of failures.
    :return: Returns a set of all slides that are sampled from or used to create training patches
    """
    parsed_folder = set()
    f = open(tracker_file, 'r')
    for line in f:
        parsed_folder.add(line.strip())
    f.close()

    return parsed_folder


def generate_patches(args, parsed_slides):
    # replace this with wsi folder
    # slide_path = '/home/ubuntu/data/TCGA_LUSC'
    slide_path = args.raw_slides_folder # '/mys3bucket/TCGA_LUSC/'
    slides_list = os.listdir(slide_path)
    mask_path = args.mask_folder if args.mask_folder else None  # '/mys3bucket/tissue-masks-CMU/'
    tar_path = args.patches_folder  # '/mys3bucket/patches'
    tmp_path = tempfile.mkdtemp()
    print(f"Temporary folder for processing: {tmp_path}")
    parsed_slides_tracker_file = args.slides_tracker

    samples = slides_list

    for slide in samples:
        slide_ID = slide.split('_')[0]
        # some slides have already been preprocessed, so skip these
        if slide_ID in parsed_slides or slide_ID in ['83183','83336','83423']:
            print ('SKIPPING')
            continue
        slide_dest = os.path.join(tar_path, slide_ID)

        slide_src = os.path.join(slide_path, slide)
        print("Reading slide " + str(slide_ID) + "...")
        try:
            image = open_slide(slide_src)
        except Exception as E:
            print(E,' SKIPPING')
            continue
        if not os.path.exists(slide_dest):
            os.mkdir(slide_dest)

        csv_path = os.path.join(slide_dest, slide_ID + ".csv")
        g = open(csv_path, 'a')

        if mask_path:
            mask_src = os.path.join(mask_path, slide_ID + ".mat")
            mask = sio.loadmat(mask_src)
            x_coords, y_coords = np.nonzero(mask['mask'])
            del mask
            gc.collect()

        x_coords = x_coords * 4
        y_coords = y_coords * 4

        count = 1
        sample_idxs = np.random.choice(np.arange(len(x_coords)),500)
        print("Randomly sampling patches ...")
        for i in sample_idxs:
            try:
                patch = image.read_region((x_coords[i],y_coords[i]),0,(256,256))
            except Exception as E:
                print(E,x_coords[i],y_coords[i])
                continue
            if keep_tile(patch,256,0.9):
                # print("Threshold passed")
                patch = patch.convert("RGB")
                outfile = os.path.join(slide_dest, "patch_" + str(count) + ".jpg")
                patch.save(outfile, 'JPEG')
                g.write(("patch_"+str(count)+","+str(x_coords[i])+","+str(y_coords[i])+"\n"))
                count += 1
            if count >=400:
                break
        g.close()
        image.close()
        print(os.path.join(tmp_path,slide))
        if os.path.isfile(os.path.join(tmp_path,slide)):
            os.remove(os.path.join(tmp_path,slide))
            print("File "+str(slide)+" flushed from tmp")
        print("Writing parsed slide to slide list")
        write_parsed_slides(slide_ID, parsed_slides_tracker_file)
        print('----------------------')


def main():
    """
    Takes in arguments from CLI for raw slides folder and folder to store the patches in
    """
    args = parse_arguments()
    parsed_slides_tracker_file = args.slides_tracker

    parsed_slides = find_parsed_slides(parsed_slides_tracker_file)
    generate_patches(args, parsed_slides)


if __name__ == '__main__':
    main()