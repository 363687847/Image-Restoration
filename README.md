# The Diplomats
## Image Restoration

Team members:
Chen Hui,
Hermeston Ryan,
Jovanovic Marko,
Mohammad Abdullah,
Patel Arjun,
Shah Arjun,
Smith Joe

The following python libraries must be installed:
wx
PIL
numpy
cv2
os
math
sys
scipy

This program was built and tested on PC using Windows 10

## How to run noise generators

All the noise generator scripts run similar to eachother. The function call in the script must
be updated in order for it to open the desired directory and add the desired noise.

Run the script by typing "python (name of noise to generate)_noise.py


## How to use the GUI

Use the original image button if you want to revert everything to original and clear all images
in the GUI except for the original image.

To use the mean filter you must only select which type of mean filter your wish to use. The only version
of the mean filter which requires user input is the contraharmonic mean filter. For this filter you must
specify the order you wish to input.

To use the order statistic filter you must only select which type of order statistic filter you wish to use.
None of the order statistic filters require any user input.

To use the adaptive filter you must only select which type of adaptive filter you wish to use. None of the 
adaptive filters require any user input.

The band pass, band reject and notch filter are all used in the same way. For all three you must specify a
lower cutoff value and a upper cutoff value. The image on the bottom left is the mask to be applied to the 
original DFT. The image on the bottom right is the original image DFT.

To perform noise sampling press the noise sampling button and click and drag to obtain the histogram of the 
desired section of the image.

To pick an image click the button and navigate to the desired file directory and select the image.

To save an image click the button and navigate to the desired file directory and save the image.

The guide to which filter to use can be found in the report.

To run file use the following command:
"python final_gui.py"

