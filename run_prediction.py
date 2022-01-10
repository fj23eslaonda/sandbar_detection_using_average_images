"""
PURPOSE   : To identify the presence of sand bar using average images
AUTHOR    : Francisco Sáez R.
EMAIL     : francisco.saez@sansano.usm.cl
V1.0      : 22/10/2021
V2.0      : 10/01/2021

This code allow identifying the presence of sand bar using a modification convolutional network propose
by Elienson et al (2021) and average images.

reference: https://www.mdpi.com/2072-4292/12/23/3953
"""


# -----------------------------------------------------------------
# 
# PACKAGES
#
# -----------------------------------------------------------------
import os
import argparse
import main
import shutil


# -----------------------------------------------------------------
# 
# INPUTS
#
# -----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Path and image size')
    parser.add_argument('--beach_path', type=str, help='beach folder')
    parser.add_argument('--main_path', type=str, default=os.getcwd(),
                        help='Duck model main path')
    parser.add_argument('--image_path', type=str, help='Input path for images')
    parser.add_argument('--output_path', type=str, help='Output path for mask')

    return parser.parse_args()


# -----------------------------------------------------------------
# 
# RUN MODEL
#
# -----------------------------------------------------------------
# Inputs
args        = parse_args()
main_path   = args.main_path
beach_path  = args.beach_path
image_path  = args.image_path
output_path = args.output_path

# Call all functions
functions = main.sandbar(main_path,
                         image_path,
                         output_path,
                         beach_path)

try:
    shutil.rmtree('.' + args.beach_path + args.output_path)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

os.system('mkdir -m 771 .' + args.beach_path + args.output_path)

# RUN MODEL
functions.run_model()
