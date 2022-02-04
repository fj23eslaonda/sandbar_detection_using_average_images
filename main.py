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
import sandbar_functions
import shutil
from pathlib import Path
from additional_functions import read_json_to_dict

# -----------------------------------------------------------------
# 
# INPUTS
#
# -----------------------------------------------------------------
parser = argparse.ArgumentParser(description='All necessary inputs and number of images to use')
parser.add_argument('--parameters', type=str,
                    help='Dictionary with all necessary inputs')
parser.add_argument('--number_img', type=int, default=False,
                    help='Number of images to use')
args = parser.parse_args()

# -----------------------------------------------------------------
# 
# RUN MODEL
#
# -----------------------------------------------------------------
# Inputs
all_inputs = read_json_to_dict(args.parameters)
all_inputs['main_path'] = Path(os.getcwd())

number_img = args.number_img

# Call all functions
functions = sandbar_functions.sandbar(all_inputs,
                                      number_img)

if (all_inputs['main_path'] / all_inputs['beach_folder'] / all_inputs['results_folder']).exists():
    shutil.rmtree(all_inputs['main_path'] / all_inputs['beach_folder'] / all_inputs['results_folder'])

(all_inputs['main_path'] / all_inputs['beach_folder'] / all_inputs['results_folder']).mkdir(mode=0o755)
# RUN MODEL
functions.run_model()
