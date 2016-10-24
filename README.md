# tf-style-transfer

Image style transfer implemented with tensorflow

<p align="center">
<img src="image_input/golden_gate.jpg" width="320"/>
<img src="image_output/golden_gate.jpg" width="320"/>
</p>

This is a personal github repository. Don't share with others untill we are ready.

### How to run

1. check imported packeges in style_model.py
2. download pretrained VGG16 nerual network file imagenet-vgg-verydeep-16.mat from http://www.vlfeat.org/matconvnet/pretrained/ and put under models folder
3. change file locations at the top of style_model.py (optional)
4. run python style_model.py using default content and style images or run python style_model.py --content content_img --style style_img

Default training finishes after 500 steps, which takes about 3 hours on mac pro cpu.

### First things you should do

1. understand basics of github - branch, pull, push, commit, etc.; use these tools
2. understand basic concepts of convolutional neural networks http://cs231n.github.io/convolutional-networks/ 
3. read tensorflow official tutorial and understand basic examples there

### Next steps

* the model is based on a paper published last year; recent method can do real-time style transfer. I will finish writing the tensorflow version in about 2-3 days.
* we need to move all training to Amazon EC2 (with GPU). Use this simple method to test out. This task includes
 * research online what's the easist way to setup AWS GPU environment (install CUDA 7.5 compatiable with tensorflow). There should be pre-set dockers/images online.
 * understand basics of tensorflow GPU training and distributed training (official tutorial has enough information)
 * modify the simple code to reduce the training time

One of you will start doing this. You two decide and let me know who is doing this task before end of Wednesday. I will join after I finish writing the new model. 

* tensorflow serving - build web services with tensorflow
 * understand and use google imange classification tensorflow serving
 * basics of dockers, kubernetes
 * build simple web service using tensorflow (we will use this after the fast model is done and trained, in about two-three weeks).
 
One of you will start doing this. 

We will have more tasks in future. 
