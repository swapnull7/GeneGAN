import argparse
import os
from openslide import open_slide
import scipy.io as sio
import numpy as np
import gc
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
import tempfile


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


def keep_tile(PILtile, tile_size, tissue_threshold):
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
    tile = np.asarray(PILtile)
    # print(tile.shape[0:2])
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
        percentage1 = tile.mean()
        check1 = percentage1 >= tissue_threshold
        '''
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
        print(check1, check2,percentage1, percentage)
        '''
        return check1
    else:
        return False


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
