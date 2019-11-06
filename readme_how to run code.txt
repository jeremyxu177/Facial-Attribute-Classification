### setup
put data and the label file 'list_attr_celeba.txt' in the same folder with the codes. we suggest to use folder '/data' for all the input data.

### crop the face
if you want to achieve better results, we highly suggest you to use the function in file 'cropface.py' to crop the face from the original input.
for function 'cropface_dataset', the inputs are folder of data, and the folder you want to output the cropped images. for function 'cropface_own', it is used
to crop any images. it can work with multiple face images. the input is only the path of the image.
### train the model
open the file 'train.py' and change the path of the training data accordingly. if you use the cropped face, change value of 'resize' in transforms.
the file will train the model and automatic save the best model.
### evaluation
run the file 'evluation.py'. the path of the model should be changed. 
the second part of evaluation.py is the test of any other images. you need to comment out if want to use it.
please change the file path if you want to have a try on other images.
it will automaticly print the image with the results.