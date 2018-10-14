# Berry-s-First-Hackathon

## Background
As more and more species face extinction everyday, the importance of tracking wildlife and documenting the number of each species in various wild environments grows as well. At the same time, the need to let them live uncaptivated remains and so does the difficulties of keeping track of their population as they run free. Human trackers may have done a good job so far, however it takes a signficantly long time to explore an entire wild area. Furthermore, the animals are also always on the move, so the possibility of missing many of them is considerably high.

With the advancement of technology in this modern era, we know that there exists means to sweep areas with drones. This can be seen in several countries, such as China, that uses drones to help police officers look for criminals and obtain their whereabouts. This is because drones can work simultaneously while communicating with each other, thus improving the overall search time significantly.

If we apply the same strategy in the wildlife areas, we will be able to generate comprehensive data of the animals in their natural habitat. By doing this, larger areas can be covered in considerably shorter timespans. In order to make use of these drones effectively, we need a method for the drone to identify and directly classify the animals, instead of humans having to manually keep track of all the animals found.

## Introduction
Tensorflow, an open source software library from google, is used in order to classify the type of animals from a given image. For the sake of simplicity, a model that has been trained by thousands of datasets available in the tensorflow is used.
A pre-trained model from official TensorFlow API that is trained with ImageNet dataset is used.

A platform that is able to handle the whole image classification process, e.g Object_detection_webcam, Object_detection_video, is then created.


## 2. Pre-Requisites
* If you do not have Python on your machine, download Python version 3.6.4. Otherwise, make sure the version you are using is compatible with the TensorFlow library. To check your current version of python installed:
```$ python --version```
Once, you have the correct version, install TensorFlow. You can clone the repository or download zip from [here](https://github.com/tensorflow/models) and follow the instructions from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

* You might need to install CUDA and cuDNN to be able to utilize the GPU as the computing device (especially for data training). You can follow the steps given [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.htm)


## 3. Installing & Running

1. After cloning the TensorFlow models shown in the [pre-requisites above](#2.-pre-requisites), copy all the files in this repository into `(cloned-tensorflow-models)/research/object_detection/`. 
2. For classify_image.py, copy it into  `(cloned-tensorflow-models)/tutorials/image/imagenet/`.
3. Download the pre-trained models from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz). After that, extract it to `(cloned-tensorflow-models)/research/object_detection/`. 

### Functions
* classify_image.py
    Default image classification created by tensorflow
* Object_detection_video.py
    Python file with the objective of classifying animals through the entire duration of an offline video
* Object_detection_image.py
    Python file with the objective of classifying animals in an image
* Object_detection_webcam.py
    Python file with the objective of classifying animals in real time

## 4. License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## 5. Author(s)
* [**Margaret Koesno**](https://github.com/maggiekoesno) - *Data Engineer*
* [**Hans Tananda**](https://github.com/hanstananda) - *AI Developer*
* [**Elbert Widjaja**](https://github.com/elbert-widjaja) - *AI Developer*

## 6. Acknowledgements

### Videos used for testing
* [Wildlife Montage](https://www.youtube.com/watch?v=GDUnGr6Ril0) by Jon Timmer
* [Cows Graze in a field](https://www.youtube.com/watch?v=fQHt1W2togc) by FarmingOnline
* [ULTIMATE Animals Video for Children](https://www.youtube.com/watch?v=bLJw9yPusak) by C Hughes

### Images used for testing
* [image1.jpg](https://commons.wikimedia.org/wiki/File:Baegle_dwa.jpg)
* [image2.jpg](https://www.nps.gov/articles/images/Image-w-cred-cap_-1200w-_-Brown-Bear-page_-brown-bear-in-fog_2_1.jpg?maxwidth=1200&maxheight=1200&autorotate=false)
* [image3.jpg](https://thenypost.files.wordpress.com/2018/02/man-eaten-by-lions.jpg?quality=90&strip=all)
* [image4.jpg](https://straightfromthehorsesmouth2you.files.wordpress.com/2011/10/sorrel-horse.jpg)

### Code reference
* [Google's TensorFlow](https://github.com/tensorflow/models) tutorial code
* [EdjeElectronic's (Evan Juras)](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) base reference for video processing using opencv

