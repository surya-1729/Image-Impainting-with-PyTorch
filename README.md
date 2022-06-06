Introduction to Image Inpainting with Deep Learning
In this report, we are going to learn how to do “image inpainting”, i.e. fill in missing parts of images precisely using deep learning. 

In this report, we are going to learn how to do “image inpainting”, i.e. fill in missing parts of images precisely using deep learning. We’ll first discuss what image inpainting really means and the possible use cases that it can cater to. Next, we’ll discuss some traditional image inpainting techniques and their shortcomings. Finally, we’ll see how to train a neural network that is capable of performing image inpainting with the CIFAR10 dataset.

Here is the brief outline of the article:

Introduction to image inpainting
Different ways of doing image inpainting
Traditional computer vision-based approaches
Deep learning-based approaches
Vanilla Autoencoder
Partial convolutions
Future directions and ending note
Github Repo →
Grab a cup of coffee and let’s dive in! This is going to be a long one.

What is image inpainting?
Image inpainting is the art of reconstructing damaged/missing parts of an image and can be extended to videos easily. There are a plethora of use cases that have been made possible due to image inpainting.

(Image inpainting results gathered from NVIDIA’s web playground)

Imagine having a favorite old photograph with your grandparents from when you were a child but due to some reasons, some portions of that photograph got corrupted. This would be the last thing you would want given how special the photograph is for you. Image inpainting can be a life savior here.

Image inpainting can be immensely useful for museums that might not have the budget to hire a skilled artist to restore deteriorated paintings.

Now, think about your favorite photo editor. Having the image inpainting function in there would be kind of cool, isn’t it?

Image inpainting can also be extended to videos (videos are a series of image frames after all). Due to over-compression, it is very likely that certain parts of the video can get corrupted sometimes. Modern image inpainting techniques are capable of handling this gracefully as well.

Producing images where the missing parts have been filled with both visually and semantically plausible appeal is the main objective of an artificial image inpainter. It’s safe enough to admit that it is indeed a challenging task.

Now, that we have some sense of what image inpainting means (we will go through a more formal definition later) and some of its use cases, let’s now switch gears and discuss some common techniques used to inpaint images (spoiler alert: classical computer vision).

Doing image inpainting: The traditional way
There is an entire world of computer vision without deep learning. Before Single Shot Detectors (SSD) came into existence, object detection was still possible (although the precision was not anywhere near what SSDs are capable of). Similarly, there are a handful of classical computer vision techniques for doing image inpainting. In this section, we are going to discuss two of them. First, let’s introduce ourselves to the central themes these techniques are based on - either texture synthesis or patch synthesis.

To inpaint a particular missing region in an image they borrow pixels from surrounding regions of the given image that are not missing. It’s worth noting that these techniques are good at inpainting backgrounds in an image but fail to generalize to cases where:

the surrounding regions might not have suitable information (read pixels) to fill the missing parts.
the missing regions require the inpainting system to infer properties of the would-be-present objects.
In some cases for the latter one, there have been good results with traditional systems. But when those objects are non-repetitive in structure, that again becomes difficult for the inpainting system to infer.

If we think of it, at a very granular level, image inpainting is nothing but restoration of missing pixel values. So, we might ask ourselves - why can’t we just treat it as another missing value imputation problem? Well, images are not just any random collection of pixel values, they are a spatial collection of pixel values. So, treating the task of image inpainting as a mere missing value imputation problem is a bit irrational. We will answer the following question in a moment - why not simply use a CNN for predicting the missing pixels?

Now, coming to the two techniques -

Navier-Stokes method: This one goes way back to 2001 (paper) and incorporates concepts from fluid mechanics and partial differential equations. It is based on the fact that edges in an image are supposed to be continuous in nature. Consider the following figure -
image.png

Along with continuity constraint (which is just another way of saying preserving edge-like features), the authors pulled color information from the surrounding regions of the edges where inpainting needs to be done.
Fast marching method: In 2004 this idea was presented in this paper by Alexandru Telea. He proposed the following:
To estimate the missing pixels, take a normalized weighted sum of pixels from a neighborhood of the pixels. This neighborhood is parameterized by a boundary and the boundary updated once a set of pixels is inpainted.
To estimate the color of the pixels, the gradients of the neighborhood pixels are used.
To have a taste of the results that these two methods can produce, refer to this article. Now that we have familiarized ourselves with the traditional ways of doing image inpainting let’s see how to do it in the modern way i.e. with deep learning.

Doing image inpainting: The modern way
In this approach, we train a neural network to predict missing parts of an image such that the predictions are both visually and semantically consistent. Let’s take a step back and think how we (the humans) would do image inpainting. This will help us formulate the basis of a deep learning-based approach. This will also help us in forming the problem statement for the task of image impainting.

When trying to reconstruct a missing part in an image, we make use of our understanding of the world and incorporate the context that is needed to do the task. This is one example where we elegantly marry a certain context with a global understanding. So, could we instill this in a deep learning model? We will see.

We humans rely on the knowledge base(understanding of the world) that we have acquired over time. Current deep learning approaches are far from harnessing a knowledge base in any sense. But we sure can capture spatial context in an image using deep learning. A convolutional neural networks or CNN is a specialized neural network for processing data that has known grid like topology – for example an image can be thought of as 2D grid of pixels. It will be a learning based approach where we will train a deep CNN based architecture to predict missing pixels.

A simple image inpainting model with the CIFA10 dataset
ML/DL concepts are best understood by actually implementing them. In this section, we will walk you through the implementation of the Deep Image Inpainting, while discussing the few key components of the same. We first require a dataset and most importantly prepare it to suit the objective task. Just a spoiler before discussing the architecture, this DL task is in a self-supervised learning setting.

Why choose a simple dataset?
Since inpainting is a process of reconstructing lost or deteriorated parts of images, we can take any image dataset and add artificial deterioration to it. For this specific DL task, we have a plethora of datasets to work with. Having said that we find that real-life applications of image inpainting are done on high-resolution images(Eg: 512 x 512 pixels). But according to [this paper]http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Generative_Image_Inpainting_CVPR_2018_paper.pdf(), to allow a pixel being influenced by the content 64 pixels away, it requires at least 6 layers of 3×3 convolutions with dilation factor 2.

Thus using such high-resolution images does not fit the purpose here. It’s a general practice to apply ML/DL concepts to toy datasets. Cutting short on computational resources and for quick implementation, we will use the CIFAR10 dataset.

Data Preparation
Certainly, the entry step to any DL task is data preparation. In our case, as mentioned we need to add artificial deterioration to our images. This can be done using the standard image processing idea of masking an image. Since it is done in a self-supervised learning setting, we need X and y (same as X) pairs to train our model. Here X will be batches of masked images, while y will be original/ground truth image.



Architecture
Inpainting is part of a large set of image generation problems. The goal of inpainting is to fill the missing pixels. It can be seen as creating or modifying pixels which also includes tasks like deblurring, denoising, artifact removal, etc to name a few. Methods for solving those problems usually rely on an Autoencoder – a neural network that is trained to copy it’s input to its output. It is comprised of an encoder which learns a code to describe the input, h = f(x), and a decoder that produces the reconstruction, r = g(h) or r = g(f(x)).

Vanilla Convolutional Autoencoder
An Autoencoder is trained to reconstruct the input, i.e. g(f(x)) = x, but this is not the only case. We hope that training the Autoencoder will result in h taking on discriminative features. It has been noticed that if the Autoencoder is not trained carefully then it tends to memorize the data and not learn any useful salient feature.

Rather than limiting the capacity of the encoder and decoder (shallow network), regularized Autoencoders are used. Usually, a loss function is used such that it encourages the model to learn other properties besides the ability to copy the input. These other properties can include sparsity of the representation, robustness to noise or to missing input. This is where image inpainting can benefit from Autoencoder based architecture. Let’s build one.

image.png
To set a baseline we will build an Autoencoder using vanilla CNN. It’s always a good practice to first build a simple model to set a benchmark and then make incremental improvements. If you want to refresh your concepts on Autoencoders this article here by PyImageSearch is a good starting point. As stated previously the aim is not to master copying, so we design the loss function such that the model learns to fill the missing points. We use mean_square_error as the loss to start with and dice coefficient as the metric for evaluation.

For tasks like image segmentation, image inpainting etc, pixel-wise accuracy is not a good metric because of high color class imbalance. Though it’s easy to interpret, the accuracy score is often misleading. Two commonly used alternatives are IoU (Intersection over Union) and Dice Coefficient. They are both similar, in the sense that the goal is to maximize the area of overlap between the predicted pixel and the ground truth pixel divided by their union. You can check out this amazing explanation here.

Wouldn’t it be interesting to see how the model is learning to fill the missing holes over multiple epochs or steps?

We implemented a simple demo PredictionLogger callback that, after each epoch completes, calls model.predict() on the same test batch of size 32. Using wandb.log() we can easily log masked images, masks, prediction and ground truth images. Fig 1 is the result of this callback. Here’s the full callback that implements this -

