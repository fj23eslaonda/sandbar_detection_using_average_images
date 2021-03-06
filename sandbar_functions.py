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
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import cv2
from additional_functions import *
from tensorflow.keras.models import model_from_json


class sandbar:

    # -----------------------------------------------------------------
    # 
    # PARAMETERS
    #
    # -----------------------------------------------------------------
    def __init__(self,
                 all_inputs,
                 number_img):

        self.main_path = all_inputs['main_path']                                # Main folder
        self.beach_path = all_inputs['main_path'] / all_inputs['beach_folder']  # Folder containing images and results
        self.image_path = self.beach_path / all_inputs['image_folder']          # Path containing the images
        self.output_path = self.beach_path / all_inputs['results_folder']       # Path to save results
        self.model_path = self.main_path / "model"                              # Model
        self.number_img = number_img                                            # Number of images used

    # -----------------------------------------------------------------
    # 
    # LOAD IMAGE NAMES LIST
    #
    # -----------------------------------------------------------------
    def load_list_img(self):
        """
        returns a list of all image names contained in image_path
        :return: list_img
        """
        list_img = sorted(os.listdir(self.image_path))

        return list_img

    # -----------------------------------------------------------------
    # 
    # LOAD DUCK MODEL
    #
    # -----------------------------------------------------------------
    def load_model(self):
        """
        Return U-Net model propose by Sáez et al. (2021)
        -----
        model is a groups layers into an object.
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model
        -----
        :return: model
        """
        # Load JSON and Create model
        json_file = open(self.model_path / 'model_1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, {"tf": tf})

        # Load weight
        model.load_weights(self.model_path / 'best_1.h5')

        return model

    # -----------------------------------------------------------------
    #
    # SELECT A WINDOW
    #
    # -----------------------------------------------------------------
    def select_window(self, name_img):
        """
        Returns the coordinates selected to identify a square image 512x512 in size,
        which the network will do the prediction.

        :param name_img:  name of initial image to set the square image
        :return: the last coordinates of the point list, e.g. the last coordinates selected.
        """
        # List to save all points selected
        points = []

        # click event function
        def click_event(event, x, y, flags, param):
            """
            Returns the coordinates you have selected.
            This function uses a CV2 tools.
            :param event: the event is when the left button is pressed.
            :param x: x-coordinate
            :param y: y-coordinate
            """
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                print_name = "x = " + str(x) + ", " + "y = " + str(y)
                cv2.putText(img, print_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                cv2.imshow("Example to select points of interest ", img)

        # Load your image of interest
        img = cv2.imread(f"{self.image_path / name_img}")
        cv2.imshow("Example to select points of interest ", img)

        # Calling the mouse click event
        cv2.setMouseCallback("Example to select points of interest ", click_event)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return points[-1]

    # -----------------------------------------------------------------
    # 
    # CREATE MATRIX FOR TEST SET
    #
    # -----------------------------------------------------------------
    def image_to_matrix(self, name_img, points):
        """
        Returns a matrix 512x512 in size to do a prediction.
        :param name_img: image name to make the prediction
        :param points: coordinates to fix the square image
        :returns: x_tst
        """
        # Create matrix
        x_tst = np.zeros((1, 512, 512, 1), dtype=np.float32)

        # Load image and crop region of interest
        image_o = cv2.imread(f"{self.image_path / name_img}", 0)
        image_f = image_o[points[1]:512 + points[1], points[0]:512 + points[0]]

        # Add new axis for network requirements
        x_img = image_f[..., np.newaxis]
        # Save images
        x_tst[0] = x_img / 255.0
        return x_tst

    # -----------------------------------------------------------------
    #
    # PLOT RESULTS
    #
    # -----------------------------------------------------------------
    def plot_results(self, points):
        """
        Returns a plot of square image predicted and prediction
        :param points: coordinates to fix square image
        :return: plot
        """
        # Load prediction
        prediction = open(self.output_path / 'prediction.json')
        prediction = dict(json.load(prediction))
        # Image names
        if self.number_img:
            list_img = self.load_list_img()[:self.number_img]
        else:
            list_img = self.load_list_img()
        # plot size
        plt.figure(figsize=(8, 6))

        for name_img in list_img:
            plt.cla()
            # Image used
            img = cv2.imread(f"{self.image_path / name_img}")
            img = img[points[1]:512 + points[1], points[0]:512 + points[0]]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Plot configuration
            plt.imshow(img, vmax=np.max(img))
            plt.xlabel("Alongshore distance, x [pixels]", fontsize=11)
            plt.ylabel("Cross-shore distance, y [pixels]", fontsize=11)
            plt.text(130, -60, 'Sand Bar Detection Problem', fontsize=11, weight="bold")
            plt.text(130, -35, 'Image: ', fontsize=11, weight='bold')
            plt.text(200, -35, name_img, fontsize=11)
            plt.text(170, -10, 'Prediction: ', fontsize=11, weight='bold')
            # If prediction is true, the points of sandbar are plotted
            if bool(prediction[name_img]):
                x_point, y_point = identify_sandbar_pts(img_gray, 'vertical', window=60, prominence=4)
                plt.scatter(x_point, y_point, c='r', marker='x')
                plt.text(280, -10, str(bool(prediction[name_img])), fontsize=11, c='b', weight='bold')
            else:
                plt.text(280, -10, str(bool(prediction[name_img])), fontsize=11, c='b', weight='bold')
            # Time between plots
            plt.pause(1)
        plt.close('all')

    # -----------------------------------------------------------------
    # 
    # RUN MODEL
    #
    # -----------------------------------------------------------------
    def run_model(self):
        """
        Main function to run the model. Load image, to do a prediction and save results.
        """
        # Inputs
        prediction = dict()
        # Image names
        if self.number_img:
            list_img = self.load_list_img()[:self.number_img]
        else:
            list_img = self.load_list_img()
        model = self.load_model()
        points = self.select_window(list_img[0])
        # ---------------------------------------------------------
        for ix, name_img in enumerate(list_img):
            print('-----------------------------------------------------------------')
            print()
            print('Prediction ' + str(ix + 1) + ' of ' + str(len(list_img)) + ' images')
            x_tst = self.image_to_matrix(name_img, points)
            y_predicted = model.predict(x_tst, verbose=True) > 0.5
            prediction[name_img] = int(y_predicted)
        # ---------------------------------------------------------
        with open(f"{self.output_path / 'prediction.json'}", 'w') as jsonfile:
            json.dump(prediction, jsonfile)
        # ---------------------------------------------------------
        query = input('Would you like to see the results?: ')
        if query:
            self.plot_results(points)
