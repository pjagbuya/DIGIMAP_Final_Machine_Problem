Framework for the video camera capture is source respectively through https://github.com/dyjdlopez/fund-dip/blob/main/modules/module_5/dip_module_05_lecture_01.py. 


The file "main.py" aims to function for both as a live camera and image scanning mechanisms. The students tinkered with two forms of formulating image analysis, (Linear Regression Models and HSV Channel Ranging). Best one employed for the main.py is HSV Channel Ranging. Its implementation consists of Live Video Scan and Scan Image Upload. 

# Limitations
- It is very important to note that the above program has been manually tested through searched Images in the Google on an image of fruit that covers most of the photo
- Unripe fruits are considered along the spectrum of green. In this case the Hue, Saturation, Value field would cover vectors (30, 20, 80) to (75, 255, 255). 
- Fruits that will produce erroneous results are naturally green ripe fruits and non green unripe fruits. Unripe Strawberry for example can yield intersects its colors with a ripe banana.

# Specifications
Using a Python 3.12 as Interpreter and Tkiner GUI, we would focus on the functions provided on main_algo, edge_removal_imginput, apply_method, adjust_gamma, and is_fruit_ripe. 

`edge_removal_imginput`, Gaussian blur and Canny edge detection were utilized for edge detection to highlight the fruit's boundaries and textures. Noise reduction techniques were fine-tuned to improve ripeness classification, as clear edges are crucial for identifying fruit maturity.


`is_fruit ripe`,  By setting the mask to ranges (30, 20, 80) to (75, 255, 255). We will use the total of the mask values to the total pixel values of the whole nonzero pixels of the image/frame. Thresholding such proportion to 0.3. "When 30% of the image is green, we would consider it as ripe/unripe."


`adjust_gamma`,  A simple slider for adjusting the brightness tone of the image. Helps with custom adjustment of the brightness of current environment.



`apply_method`,  Puts text over the image and identifies itself to be either "Ripe" or "Unripe" that is mainly aided by `is_fruit ripe`
