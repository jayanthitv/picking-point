# Mango_picking_point_positioning

# Introduction.
This is a mango picking point prediction system based on YOLOv8n target detection and YOLOv8n-seg instance segmentation, designed for four mango varieties, including Guiqi, Ao Mango, Jinhuang and Tainong. The function of the system is to identify the picking points of mango fruit stalks in the mango images, along with the statistics of the number of fruits, the number of fruit stalks, the prediction time and the selection of model weights. The coordinates of the picking points are used to guide the fruit picking work of the collecting robot, which plays an important role in accurate picking.

# Software Architecture
Dataset from {https://doi.org/10.17632/hykdyc24xp.2 (original image datasets) and https://doi.org/10.17632/hn866dtzrr.1  (enhanced image datasets)}


# Installation Tutorial
To run this project, the dependencies you need to install and their versions are given in requirements.txt.

# Instructions for use
1. ui_main.py file is used to visualise the window operation; predict_main.py file can be used to predict the whole folder of pictures directly; step1_main.py file is used to predict the picking point for a single picture.

2. The weight folder arranges the target detection model weights, and the weights can be selected under the ui_main.py file. And the model weights  are placed in the weight file , if you need to replace them, please do it in hrnet.py under hrnet file.

3. The target detection results and language segmentation results are in the runs file, while the final prediction result graph is saved in the results file.

# Remarks
1. After installing according to "pip install ultraltyics", if the programme reports that some modules do not exist, you can directly copy the ultraltyics folder in the programme to cover the original folder in the anaconda environment.



