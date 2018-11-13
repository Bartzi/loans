# LoANs

Code for the AMV18 Paper "LoANs: Weakly Supervised Object Detection with Localizer Assessor Networks".
You can read a Preprint on [Arxiv (TODO)](TODO).

This page contains information on everything you need to know to train a model using our weakly supervised approach.
In order to successfully train a model you'll need to following ingredients:
1. a specific type of object that you want to be localized.
2. Images that contain this object (you might also take a video of some minutes length and extract each frame).
3. Some background images on wich the object might appear.
4. Some template images of you object from different view points.
5. The code from this repository.
6. A decent GPU.
7. ...
8. Profit.

This README will guide you through the process of training a model, using our approach.
We will train a model that can localize a figure skater on a given image and apply the trained model
on a test video. While going through each step, we will provide further information on each option
our scripts provide.

# Training a model for localizing figure skaters

In order to successfully train a model for localizing figure skater,
we will first need to setup the code of this repository on your computer.
Scond, we will gather some training data.
We will then use this data to train a model.
After training the model, we will use some visualization scripts to have
a look at the results of our training.

## Setup

In order to make full usage of the code in this repository you will need
to perform the following steps:
1. Install a Python Version `>= 3.6`. It is important to have a Python
version that is greater than **3.6**, because the code won't work with
older python versions.
2. Create a new `virtualenvironment` for this code. You can use the
excellent [Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
for this purpose.
3. Install all requirements with `pip install -r requirements.txt`.
4. Make sure you have `CUDA`, `cuDNN` and if you like also `NCCL` installed.
4. Sit back and have a coffee/tea, while `cupy` compiles.
5. If you want to follow this README, it makes sense to install `youtube-dl`.

## Data Preparation

We will need to prepare two kinds of data for training the model.
As the overall model consists of two independent neural networks, we
need training data for each of these networks.

### Train Data for training the localizer

Getting the data for the localizer is fairly easy. You basically have to
go to YouTube and find some figure skating videos.
We've already prepared some videos that you can download and use for training:
- [Alina Zagitova](https://www.youtube.com/watch?v=TlXCk1LDlC0) (Olympic Games 2018)
- [Yuzuru Hanyu](https://www.youtube.com/watch?v=23EfsN7vEOA) (Olympic Games 2018)
- [Yulia Lipnitskaya](https://www.youtube.com/watch?v=ke0iusvydl8) (Olympic Games 2014)
- [Jason Brown](https://www.youtube.com/watch?v=J61k2XjRryM) (US Open 2014)

For validating our model, we will use the following video:
- [Yuna Kim](https://www.youtube.com/watch?v=hgXKJvTVW9g) (Olympic Games 2014)

You can download each video with `youtube-dl` and save in a directory on your PC.
We will assume that we downloaded all videos into a directory called `videos` that is
located directly in the root directory of this repository. Make sure to save the
validation video in a different spot!

We will now extract every single frame from the downloaded videos and save those
in the folder `train_data/localizer`. In order to do this, you can use
the script `extract_frames_from_video.py` that you can find in the `video_analysis`
directory.

The usage of this script is as follows:
```
python video_analysis/extract_frames_from_video train_data/localizer -i videos/* -r 512

train_data/localizer: output directory
-i: all videos you want to use for training
-r: Resize all extracted frames in such a way that the longest side has a
dimension of 512px (mainly useful for saving space and speeding up the training
later on)
```

The directory `training_data/localizer` will contain a subdirectory for each analyzed video
and also a file `gt.csv` that lists all extracted images.


