# Irish Red Squirrel CNN Classifier

This project used a CNN to identify wildlife camera images containing red squirrels and to seperate them from other images placing them in a seperate folder. The CNN was trained on a subset of approximately 750 images and was then applied to the whole dataset of around 6,000 images to filter images containing red squirrels automatically. This project uses transfer learning and is based on the Tensorflow tutorial https://www.tensorflow.org/tutorials/images/transfer_learning. 

The required packages for this project can be installed using the command
pip install -r requirements.txt. Environment variables are used to specify the directories for training files, target directory to be filtered and where the filtered images are to be copied to.