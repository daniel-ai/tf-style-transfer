# tf-style-transfer

Tensorflow implementation of a collection of style transfer methods

 <p align="center">
<img src="image_input/c_golden_gate.jpg" width="400"/>
<img src="image_input/style.jpg" width="200"/>
</p>

 1. slow single image method: optimize each pixel value of an image to preserve content features from the original content image and style features from a style image
 
 _Image Style Transfer Using Convolutional Neural Networks_
 

 <p align="center">
<img src="image_output/golden_gate_single_image.jpg" width="400"/>
</p>

 2. fast single style method: optimize a image transfer neural network for each style image to preserve similar content features and style features.
 
 _Perceptual Losses for Real-Time Style Transfer and Super-Resolution_
 
 <p align="center">
<img src="image_output/golden_gate_single_style.jpg" width="400"/>
</p>
 
 3. fast general style method: train a genera image transfer neural network with only normalization layers optimized for each style image to preserve similar content features and style features.

### How to run

 1. Install packeges tensorflow, numpy, scipy, PIL
 2. download pretrained VGG16 nerual network file imagenet-vgg-verydeep-16.mat from http://www.vlfeat.org/matconvnet/pretrained/ and put under models folder
 3. for method 2&3, preprare a training set of content images (http://mscoco.org/dataset/#download) and resize all images to a standard size ([256, 256]).

To train a model:
```
python slow_single_image.py --content your_content_image --style your_style_image --vggfile downloaded_vgg_mat_file --max_iteration n
python fast_single_style.py --style your_style_image --train_dir training_image_dir --vggfile downloaded_vgg_mat_file --max_iteration n
python fast_general_style.py --style_dir your_style_image_dir(s_*.jpg) --train_dir training_image_dir --vggfile downloaded_vgg_mat_file --max_iteration n
```
To stylize an image for method 2&3:
```
python fast_single_style.py --stylize --content_dir your_content_image_dir(c_*.jpg) --model your_saved_session_file
python fast_general_style.py --stylize --content_dir your_content_image_dir(c_*.jpg) --model your_saved_session_file
```

Check the code for more options.

