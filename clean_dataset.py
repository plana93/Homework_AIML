import os
import shutil
TO_DELETE = ".DS_Store"
RGB_FOLDER = 'rgb'
RGB_PREFIX = 'rgb'
MMAP_FOLDER = 'mmaps'
MMAP_PREFIX = 'map'


def adjust_maps(directory):
    to_create = []
    # find all sub folders of directory
    sub_folders = [el for el in os.listdir(directory) if os.path.isdir(os.path.join(directory, el))]
    if RGB_FOLDER not in sub_folders:
        for sub_f in sub_folders:
            next_folder = os.path.join(directory, sub_f)
            adjust_maps(next_folder)
    else:
        rgb_frames = os.listdir(os.path.join(directory, RGB_FOLDER))
        mmap_frames = os.listdir(os.path.join(directory, MMAP_FOLDER))
        if len(rgb_frames) != len(mmap_frames):
            rgb_filenums = sorted([frame.split(RGB_PREFIX)[1] for frame in rgb_frames])
            map_filenums = sorted([frame.split(MMAP_PREFIX)[1] for frame in mmap_frames])
            for i, rgb_file in enumerate(rgb_filenums):
                if rgb_file not in map_filenums:
                    if i == 0:
                        # the file will be replaced with the one to the right (i+1)
                        to_create.append((rgb_filenums[i+1], rgb_filenums[i]))
                    elif i == len(rgb_filenums) - 1:
                        # replace the file with the one to the right (i-1)
                        to_create.append((rgb_filenums[i-1], rgb_filenums[i]))
                    else:
                        # replace the file mixing the one to the right and left
                        to_create.append((rgb_filenums[i - 1], rgb_filenums[i + 1], rgb_filenums[i]))
        for copy in to_create:
            source_filename = os.path.join(directory, MMAP_FOLDER, MMAP_PREFIX+copy[0])
            copy_filename = os.path.join(directory, MMAP_FOLDER, MMAP_PREFIX+copy[1])
            print("duplicating "+source_filename.split("frames2")[1]+" as "+copy_filename.split("frames2")[1]+"....")
            shutil.copy(source_filename, copy_filename)


def clean_data(directory, depth=0):
    elements = os.listdir(directory)
    print(directory)
    if TO_DELETE in elements:
        print("removing ..  "+os.path.join(directory, elements[elements.index(TO_DELETE)]))
        os.remove(os.path.join(directory, elements[elements.index(TO_DELETE)]))
    if depth < 2:
        for el in elements:
            d = os.path.join(directory, el)
            if os.path.isdir(d):
                clean_data(d, depth+1)


def clean_data_old(directory):
    print(directory)
    elements = os.listdir(directory)
    if TO_DELETE in elements:
        print("removing ..  "+os.path.join(directory, elements[elements.index(TO_DELETE)]))
        os.remove(os.path.join(directory, elements[elements.index(TO_DELETE)]))
    for el in elements:
        d = os.path.join(directory, el)
        if os.path.isdir(d):
            clean_data(d)