import sys
import argparse
import os
import random
from openslide import open_slide, ImageSlide
import scipy.io as sio
import pdb
import numpy as np
import gc
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
    #od = -np.log10(tile/255 + 1e-8)
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


def generate_patches(wsi_folder, patches_folder, parsed_slides):
    # replace this with wsi folder
    # slide_path = '/home/ubuntu/data/TCGA_LUSC'
    slide_path = '/mys3bucket/TCGA_LUSC/'
    slides_list = os.listdir(slide_path)
    mask_path = '/mys3bucket/tissue-masks-CMU/'
    tar_path = '/mys3bucket/patches'
    tmp_path = '/tmp/pathai-cmu-capstone/TCGA_LUSC'

    samples = slides_list
    # print(len(samples))
    # print(samples)
    for slide in samples:
        print(slide)
        # input()
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
        mask_src = os.path.join(mask_path, slide_ID + ".mat")
        mask = sio.loadmat(mask_src)
        x_coords, y_coords = np.nonzero(mask['mask'])
        del mask
        gc.collect()

        x_coords = x_coords * 4
        y_coords = y_coords * 4
        # nonzero_coords = list(zip(x_coords, y_coords))
    	# pdb.set_trace()
        count = 0
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
                outfile = os.path.join(slide_dest, "patch_" + str(count+100) + ".jpg")
                patch.save(outfile, 'JPEG')
                g.write(("patch_"+str(count+100)+","+str(x_coords[i])+","+str(y_coords[i])+"\n"))
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
        write_parsed_slides(slide_ID)
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
        help='folder contatining patches generated from slides')

    args = parser.parse_args()
    return args

def write_parsed_slides(slideID):
    f = open('./slides_new.list', 'a')
    f.write(str(slideID)+"\n")
    f.close()
    return

def find_parsed_slides(folder):
    parsed_folder = set()
    f = open('./slides_new.list', 'r')
    for line in f:
        parsed_folder.add(line.strip())
    f.close()
    # print(parsed_folder)
 
    # for filename in os.listdir(folder):
    #    basename, extension = os.path.splitext(filename)
    #    parsed_folder.add(basename.split('_')[-1])
    return parsed_folder


def main():
    # args = parse_arguments()
    # wsi_slides_folder = args.raw_slides_folder
    wsi_slides_folder = '/mys3bucket/TCGA_LUSC/'
    patches_folder = '/mys3bucket/patches/'

    parsed_slides = find_parsed_slides(patches_folder)
    generate_patches(wsi_slides_folder, patches_folder, parsed_slides)


if __name__ == '__main__':
    main()
