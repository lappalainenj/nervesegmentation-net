**Ultrasound Nerve Segmentation using CNNs**

---

## Folder structure

Please take care that your project folder is structured as follows:

project/
     ../data
            ../images
            ../targets
     ../nerve_segmentation
                        

where in **project/data/images** are the ultrasonic pictures and in **project/data/targets** the masks from 'train.zip'. 
The folder **project/nerve_segmentation** is automatically initialized by pulling with git. It contains all the program code.

The program structure must be strictly maintained, whereas new folder can be added. 

## Main jupyter notebook

The notebook **nerve_segmentation.ipynb** should finally contain a structured representation of our results, similar to the exercise notebooks that we were working on in during the lecture period.

In order to use this notebook for the development of new features and trying things out, please **make** an independent **copy** of it first.


## Description from Kaggle
Source: https://www.kaggle.com/c/ultrasound-nerve-segmentation

"Even the bravest patient cringes at the mention of a surgical procedure. Surgery inevitably brings discomfort, and oftentimes involves significant post-surgical pain. Currently, patient pain is frequently managed through the use of narcotics that bring a bevy of unwanted side effects.

This competition's sponsor is working to improve pain management through the use of indwelling catheters that block or mitigate pain at the source. Pain management catheters reduce dependence on narcotics and speed up patient recovery.

Accurately identifying nerve structures in ultrasound images is a critical step in effectively inserting a patient’s pain management catheter. In this competition, Kagglers are challenged to build a model that can identify nerve structures in a dataset of ultrasound images of the neck. Doing so would improve catheter placement and contribute to a more pain free future."

## Description of the Data

The task in this competition is to segment a collection of nerves called the Brachial Plexus (BP) in ultrasound images. You are provided with a large training set of images where the nerve has been manually annotated by humans. Annotators were trained by experts and instructed to annotate images where they felt confident about the existence of the BP landmark.

Please note these important points:

The dataset contains images where the BP is not present. Your algorithm should predict no pixel values in these cases.
As with all human-labeled data, you should expect to find noise, artifacts, and potential mistakes in the ground truth. Any individual mistakes (not affecting the broader integrity of the competition) will be left as is.
Due to the way the acquisition machine generates image frames, you may find identical images or very similar images.

File descriptions

**train** contains the training set images, named according to subject_imageNum.tif. Every image with the same subject number comes from the same person. This folder also includes binary mask images showing the BP segmentations.


