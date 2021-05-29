from re import T
import scipy, numpy, shutil, os, nibabel, glob
import sys, getopt

from tensorflow.python.keras.backend import print_tensor
from utils.settings import *
import imageio


def main():
    inputfile_paths = []
    outputfile_paths = []
    for root, _, files in os.walk(ADNI_RAW_DATA_DIR):
        for filename in files:
            if filename.endswith(".nii"):
                filepath = os.path.join(root, filename)
                inputfile_paths.append(filepath)
                file_output_path = filepath[::-1].replace("/raw/"[::-1], "/transformed/"[::-1], 1).replace(".nii"[::-1], ".png"[::-1])[::-1]
                outputfile_paths.append(file_output_path)
    
    
    for inputfile, outputfile in zip(inputfile_paths, outputfile_paths):
        print(inputfile, outputfile)
        transform(inputfile, outputfile)
        
        

def transform(inputfile, outputfile):
    

    # set fn as your 4d nifti file
    image_array = nibabel.load(inputfile).get_data()
    
    

    # if 4D image inputted
    if len(image_array.shape) == 4:
        
        # set destination folder
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
            print("Created ouput directory: " + outputfile)

        print('Reading NIfTI file...')

        total_volumes = image_array.shape[3]
        total_slices = image_array.shape[2]

        # iterate through volumes
        for current_volume in range(0, total_volumes):
            # iterate through slices
            for current_slice in range(0, total_slices):
                
                # rotate or no rotate
                data = image_array[:, :, current_slice, current_volume]
                #alternate slices and save as png
                print('Saving image...')
                image_name = inputfile[:-4] + "_t" + "{:0>3}".format(str(current_volume+1)) + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
                
                imageio.imwrite(image_name, data)
                print('Saved.')

                #move images to folder
                print('Moving files...')
                src = image_name
                shutil.move(src, outputfile)
            

        print('Finished converting images')

    # else if 3D image inputted
    elif len(image_array.shape) == 3:
        print("/".join(outputfile.split("/")[:-1]))
        print(outputfile.split("/")[-1])
        
        
        
        # set destination folder
        if not os.path.exists("/".join(outputfile.split("/")[:-1])):
            os.makedirs("/".join(outputfile.split("/")[:-1]))
            print("Created ouput directory: " + "/".join(outputfile.split("/")[:-1]))

        print('Reading NIfTI file...')

        total_slices = image_array.shape[2]
        
        # iterate through slices
        
        data = image_array[:, :, 100]
        #alternate slices and save as png
        image_name = outputfile
        imageio.imwrite(image_name, data)

        #move images to folder
        src = image_name
        #shutil.move(image_name, "/".join(outputfile.split("/")[:-1]))

        print('Finished converting images')
    else:
        print('Not a 3D or 4D Image. Please try again.')

# call the function to start the program
if __name__ == "__main__":
   main()