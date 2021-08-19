# Detecting Guns In A Tarantino Movie
Started out as an image classification project using Transfer Learning on a custom data of a popular TV show, Westworld but ended up as an object detector using yolov5 on a much larger dataset.

## Image Classification
Image classification is where a computer can analyse an image and identify the ‘class’ the image falls under. (Or a probability of the image being part of a ‘class’.) 

Gun-violence is violence committed with the use of a gun (firearm or small arm) and individuals carrying firearms in public places are a strong indicator of dangerous situation. There are many solutions proposed to this problem ranging from stricter gun laws to Psychiatry. My main goal is to add in a way to not let situation go out of control by using deep learning (CNN) to create a model that will be able to detect guns and people carrying firearms. Here the aim is to train a neural network to classify images as either having gun violence or not

### Dataset
Westworld is a popular HBO series renowned for it's violence and deaths of its characters. I used OpenCV to capture the videos (episodes) and turned them into frames. The whole code is present in the data_collection notebook.

```
count = 0
for i in range(1, 11):
    videoFile = f"Westworld.S01E{i}.mkv"
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename ="frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()
print("Done!")
```

Next, I manually went through all of the images to seperate them as per my objective and with the help of python libraries such as os, shutil and random, I transformed the data into training and test set.
Also added in this kaggle dataset as one of the solutions to my largely imbalanced dataset - 

https://www.kaggle.com/issaisasank/guns-object-detection

### Transfer Learning
Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

Selected Model :-

VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. VGG16 significantly outperforms the previous generation of models in the ILSVRC-2012 and ILSVRC-2013 competitions.

### Conclusion
Used Flask to deploy my model to web based interface

<img src="https://github.com/King-Kapoor/Detecting-guns-in-a-Tarantino-movie/blob/main/output.gif" width="450">


## Object Detection
Object detection is a computer vision task that requires object(s) to be detected, localized and classified. In this task, first we need our machine learning model to tell if any object of interest is present in the image. If present, then draw a bounding box around the object(s) present in the image. In the end, the model must classify the object represented by the bounding box.


### Dataset
Besides the above mentioned dataset, I scraped more high quality images from the website - Internet Movie Firearms Database, IMFDB for short (http://www.imfdb.org/wiki/Main_Page) which has an extensive database of images of firearms from many movies. For my purpose I picked up movies from Netflix, HBO and all Quentin Tarantino movies (except Django Unchained on whom I later tested my model on). I used Beautiful Soup and requests for that task. The whole code is present in the data_collection notebook.


### Data Labelling
I used an online labelling site - make-sense (https://www.makesense.ai/) which is a free to use online tool for labelling photos that just requires a browser and not require any complicated installation.

### Yolov5
YOLO is an abbreviation for the term 'You Only Look Once'. This is an algorithm that detects and recognizes various objects in a picture (in real-time).

Each of the versions of YOLO kept improving the previous in accuracy and performance. Then came YOLOv4 developed by another team, further adding to performance of model and finally the YOLOv5 model was introduced by Glenn Jocher in June 2020. This model significantly reduces the model size (YOLOv4 on Darknet had 244MB size whereas YOLOv5 smallest model is of 27MB).


### Models
The models I obtained after training are in here - https://drive.google.com/drive/folders/1q26YxTaW-LGYtMsOxjz_D3zNwVcmEofG?usp=sharing

### Conclusion

Reulted Images - 

<img src="https://github.com/King-Kapoor/Detecting-guns-in-a-Tarantino-movie/blob/main/yolo_voilence1.jfif">

<img src="https://github.com/King-Kapoor/Detecting-guns-in-a-Tarantino-movie/blob/main/yolo_voilence2.jfif">


And finally, Detecting guns in the movie Django Unchained - 

https://drive.google.com/drive/folders/1y_MCnnaNQu8kxAC6z8Tlt5X3-_4Oz_lz?usp=sharing





