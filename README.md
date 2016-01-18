# DeepMultiBox_ObjectDetection
Using DNN based object detection to identify salient objects in a class-agnostic manner to identify characters in animation movies

Step 1) - Install caffe2, download pre-trained object detection files and related directories as described here - https://github.com/google/multibox/blob/master/multibox.ipynb

Step 2) - Build a directory of key images or imgaes with objects of interest (If you are subsampling a full movie by say a downsampling ration of N/10; use ffmpeg -i movie.avi -r 2.398 %04d.ppm)

Step 3) - Edit paths and run detect_deep_multibox.py which outputs a neat json file and then display or save confident objects with the other script available.
