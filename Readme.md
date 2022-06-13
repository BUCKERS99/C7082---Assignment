# C7082 - Computer Vision to detect blight in potato leaves
The repository for the C7082 Machine Learning and AI assignment

# Background
Potato blight is a disease that has a historically noticeable presence and has caused food shortages in a
range of countries; the most harrowing example is the Irish potato famine in the mid 1800’s (O’Neill, 2009).
The current methods of controlling the disease which is caused by the fungus Phytophthora infestans (late
blight) and Alternaria Solani (early blight), is to employ a spray program that requires field wide chemical
treatment of crops: up to twice a week, depending on the chemical being used (Leesutthiphonchai et al.,
2018). With the world population expected to increase to around 9.1 billion by 2050 (Cohen, 2001), the need
for secure food production in the future is important if we are to supply the increase in people. With genetic
modification (GM) currently not an option in the UK, a line of potatoes with full resistance to blight would
not be available for at least 30 years due to the time it takes to breed in a gene (Ceccarelli, 2015; Haverkort
et al., 2009).
There have been observations of reduced efficacy in the chemical control of blight when using active ingredients such as fluazinam (Schepers et al., 2018). This can be attributed to the overuse of a chemical strategy
and a build-up of resistance in the fungus. The active ingredients used in the control of blight has deadly
and long-lasting effects on the environment and is highly toxic to humans (EFSA, 2008). It has been seen
to cause severe damage to aquatic habitats and this can easily occur in the UK where chemical are applied
in adverse conditions and creates run-off.
For the reasons of environmental sustainability and with the cost of applications adversely affecting the
producers’ profit margins, the implementation of blight recognition using a machine learning technique
could provide a future proof, precision application method of blight control. For example, the adaption of
a drone to carry out fungicide application with a camera attached to recognise plants that have a blight
infection could reduce the amount of fungicide used, increasing profit margins, and prevent the field wide
application of the pesticide which will reduce the chance of run-off.
As the data set used in this assignment is made up of images, the decision was made to use Convolutional
Neural Networks (CNN) to analyse and classify the images. CNNs have been proven in their use for classification tasks and provide a high accuracy, when tuned correctly, on validation data sets (Yoo, 2015). The objectives of this assignment are:

1. Use different pre-trained models to identify an image of a healthy leaf, a late blight infected leaf and
an early blight infected leaf
2. Tune the model that produced the lowest validation accuracy, in the first run, to see if it is possible to
outperform the initial validation accuracy of the best performing model (first run).

# Methods
## Data
The dataset was taken from Kaggle and contains images of healthy potato leaves (Kaggle, 2021a) potato
leaves that are infected with early blight (A. solani) and potato leaves that are infected with late blight (P.
infestans). Within this data set there were 2152 files split as follows:

* Early blight - 1000 images - JPG files
* Late blight - 1000 images - JPG files
* Healthy - 152 images - JPG files

It was decided that there being 15% of amount of healthy potato leaf images compared to the diseased images
had the potential to cause problems when training the model; this having been considered, the decision was
made to collect more healthy potato leaf images. This was done using Kaggle where a further 816 images
were taken from the training folder (Kaggle, 2021b).

* Healthy - 968 images - JPG files

The image size for most of the images was 256x256 pixels, to make sure that the images were inputted into
the model with a consistent size it was decided that they should be rescaled when necessary.

## File re-naming
It was found that the names of all the files were a random mix of letters and numbers. The decision
was made to rename the files and order them numerically depending on the category they were from.
This was more for user benefit as it does not affect the machine learning model. “00fc2ee5-729f-4757-8aeb65c3355874f2___RS_HL 1864.JPG” was renamed to “Healthy_1.JPG” and this was done using consecutive
numbers. This was repeated for late blight leaves, early blight leaves and healthy leaves.

## File sorting
The total images were split into three categories: test, train, and validation images and an 80/10/10 for
train/test/validate was used. Table 1 shows the number of files in each folder for each category

|              | Test | Train | Validate |
|--------------|------|-------|----------|
| Healthy      | 97   | 97    | 772      |
| Late Blight  | 100  | 100   | 800      |
| Early Blight | 100  | 100   | 800      |

# Models
## VGG16

VGG16 has been used in many image classification problems; notably in a study conducted by Rangarajan
and Purushothaman (2020) it was used to detect disease in eggplant images. VGG16 is designed for image
classification with 13 stacked convolution layers. Each layer extracts features depending on the disease image
it is given. A softmax function is used on the output in order for a probability score to be made to each
class. An example of the architecture of VGG16 can be seen in Figure 1.

![](https://github.com/BUCKERS99/C7082---Assignment/blob/main/Images/VGG.PNG)

*Figure 1: VGG16 Architecture*

The images were rescaled to 255x255 and were inputted into the model as red, green, blue (rgb) scale, as was
the same for all the initial run of the models. The layers were frozen to prevent each layer being updated
during the training of the model. To save on computing power and time, 10 epochs were chosen with 50
steps per epoch.
A training accuracy of 98.4% and a validation accuracy of 80.1% was recorded after the initial running of
the model.

## Xception
In comparison, both Xception and VGG16 are trained on the ImageNet (2021); however, Xception is a CNN
that is said by its creator Chollet (2017) to outperform both VGG16 and Inception v3 due to its depth wise
separable convolutions. For the image dataset used in this instance the learning layers were again frozen and
10 epochs with 50 steps per epoch were chosen in order to have a direct comparison to the other models. A
training accuracy of 98.4% and a validation accuracy of 79.8% was recorded after the running of the model
which was lower than the VGG16 model.

## Inception V3
Inception v3 is a pre-trained model that was also trained on the ImageNet data set which allows for a good
comparison between the three models (Szegedy et al., 2015). It is based on the idea that each layer is an
inception layer where each layer’s output is filtered into the input of the next layer, shown in Figure 2.

![](https://github.com/BUCKERS99/C7082---Assignment/blob/main/Images/inceptionv3.PNG)

*Figure 2: Inception v3 architecture*

This has a negative effect of increasing the computational power needed to run the model, but should, in
theory increase the accuracy.
A training accuracy of 76.9% and a validation accuracy of 68.6% was achieved using the inception pre-trained
model with the same number of epochs and steps per epoch as the previous models. As the inception v3
model achieved the lowest overall accuracy, it was taken forward and tuned to see if it could meet, or exceed,
the benchmark of 80.1% posed by the VGG16 model. The comparison of the accuracy can be seen in figures
3 and 4.

![](https://github.com/BUCKERS99/C7082---Assignment/blob/main/Plots/Inception_model_accuracy.PNG)

*Figure 3: Inception v3 model accuracy*

![](https://github.com/BUCKERS99/C7082---Assignment/blob/main/Plots/VGG16_model_accuracy.PNG)

*Figure 4: VGG16 model accuracy*

# Tuning

## 1
Increasing the learning rate from 0.0001 to 0.0005 produced a training accuracy of 91.4% and a validation
accuracy of 84.9%. This is already 5% more accurate on the validation set than the initial Inception v3 run.
Tuning the learning rate from 0.0001 to 0.0005 has beaten the benchmark validation accuracy set by VGG16
by 4.8%; as this has completed the objective a new objective was set:
1. How close to 100% validation accuracy can we get the model to be?

## 2
To reduce the computing power, the next model contained the early stopping function. This prevented the
model running when there was little, or no improvement seen in the validation accuracy. Also contained in
the second model, was the activation change from relu to softmax. Having run this, the training accuracy
was 51.3% and the validation accuracy was 54.9% however, the early stopping call back stopped the model
after 2 epochs.

## 3
For the third change, the batch size was changed from 32 to 64, for this to work the number of steps per
epoch were decreased to 25 steps per epoch. The results of this test produced a training accuracy of 82.4%
and a validation accuracy of 78.9%.

## 4
Increasing the learning rate further to 0.001 and returning the batch size to 32 and the steps per epoch to
50 saw a training accuracy of 83.6% and a validation accuracy of 81.4%. the closing of the gap between test
and validation accuracy shows that the model is over fitting less, which is a positive discovery.

## 5
For the fifth version of the model the number of steps per epoch were increased from 50 to 75 and the training
accuracy was recorded at 90.9% and the validation accuracy at 85.1%.

## 6
The sixth model required the image size to be increased to 200,200 and run with the learning rate of 0.0005
and 75 steps per epoch. This gave a training accuracy of 87.5% and a validation accuracy of 83.8%.

## 7
For the seventh model the dense layers in the relu activation were increased from 1024 to 2048, with the
image size returning to the original: 150x150. This recorded a training accuracy of 96.7% and a validation
accuracy of 85.9%.

# Results
The model that achieved the highest validation accuracy was the final model, the seventh tune. The accuracy
on the training data set was 96.7% and the accuracy on the validation data set was 85.9%. There is over
10% difference between these accuracies which suggest some element of overfitting. Figure 5 and 6 show the
accuracy and the loss associated with the model ran.

![](https://github.com/BUCKERS99/C7082---Assignment/blob/main/Plots/inception_model_7_accuracy.PNG)

*Figure 5: Tune 7 accuracy*

![](https://github.com/BUCKERS99/C7082---Assignment/blob/main/Plots/inception_model_7_loss.PNG)

*Figure 6: Tune 7 cost*

The model cost reduced throughout when training the model, however it increased during the testing from
0.58 to 2.7.

The table below outlines the validation and training accuracies throughout tuning the model. It shows that
the first change to the original model produced results that outperformed the benchmark. This required a
new objective to be set, the closest the model got to a 100% accuracy using the Inception V3 pre-trained
model was 85.9%.

|            | Training Accuracy (%) | Validation Accuracy (%) |
|------------|-----------------------|-------------------------|
| BENCHMARK  | 98.4                  | 80.1                    |
| Base Model | 76.9                  | 68.6                    |
| Tune 1     | 91.4                  | 84.9                    |
| Tune 2     | 51.3                  | 54.9                    |
| Tune 3     | 82.4                  | 78.9                    |
| Tune 4     | 83.6                  | 81.4                    |
| Tune 5     | 90.9                  | 85.1                    |
| Tune 6     | 87.5                  | 83.8                    |
| Tune 7     | 96.7                  | 85.9                    |
|            |                       |                         |


# Files 
[PDF of report](https://github.com/BUCKERS99/C7082---Assignment/blob/main/17239400_C7082.pdf)

[Model coding](https://github.com/BUCKERS99/C7082---Assignment/blob/main/C7082_assignment_coding.ipynb)

[Image folder creation coding](https://github.com/BUCKERS99/C7082---Assignment/blob/main/Train_test_val_folder_coding.ipynb)

[Image renaming coding](https://github.com/BUCKERS99/C7082---Assignment/blob/main/C7082_file_rename_coding.ipynb)

[Plots folder](https://github.com/BUCKERS99/C7082---Assignment/tree/main/Plots)

[Images folder](https://github.com/BUCKERS99/C7082---Assignment/tree/main/Images)

[Potato leaf images for analysis](https://github.com/BUCKERS99/C7082---Assignment/tree/main/potato_leaf_pics)

