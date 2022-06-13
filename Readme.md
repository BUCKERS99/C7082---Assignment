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






# Files 
[PDF of report](https://github.com/BUCKERS99/C7082---Assignment/blob/main/17239400_C7082.pdf)

[Model coding](https://github.com/BUCKERS99/C7082---Assignment/blob/main/C7082_assignment_coding.ipynb)

[Image folder creation coding](https://github.com/BUCKERS99/C7082---Assignment/blob/main/Train_test_val_folder_coding.ipynb)

[Image renaming coding](https://github.com/BUCKERS99/C7082---Assignment/blob/main/C7082_file_rename_coding.ipynb)

[Plots folder](https://github.com/BUCKERS99/C7082---Assignment/tree/main/Plots)

[Images folder](https://github.com/BUCKERS99/C7082---Assignment/tree/main/Images)

[Potato leaf images for analysis](https://github.com/BUCKERS99/C7082---Assignment/tree/main/potato_leaf_pics)

