from matplotlib import pyplot
import numpy as np
import os
from skimage import io, transform
import sys, json
# Make sure that you set this to the location your caffe2 library lies.
caffe2_root = '/opt/caffe2/'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

print caffe2_root
# After setting the caffe2 root path, we will import all the caffe2 libraries needed.
from caffe2.proto import caffe2_pb2
from pycaffe2 import core, net_drawer, workspace, visualize

multibox_data_dir = '/opt/caffe2/multibox/'

# net is the network definition.
net = caffe2_pb2.NetDef()
net.ParseFromString(open(os.path.join(multibox_data_dir, 'multibox_net.pb')).read())

# tensors contain all the parameters used in the net.
# The multibox model is relatively large so we have stored the parameters in multiple files.
import glob
file_parts = glob.glob(multibox_data_dir+"multibox_tensors.pb.part*")
file_parts.sort()
tensors = caffe2_pb2.TensorProtos()
tensors.ParseFromString(''.join(open(f).read() for f in file_parts))

# Note that the following line hides the intermediate blobs and only shows the operators.
# If you want to show all the blobs as well, use the commented GetPydotGraph line.
#graph = net_drawer.GetPydotGraphMinimal(net.op, name="multibox", rankdir='TB')
#graph = net_drawer.GetPydotGraph(net.op, name="inception", rankdir='TB')

#print 'Visualizing network:', net.name
#display.Image(graph.create_png(), width=200)

DEVICE_OPTION = caffe2_pb2.DeviceOption()
# Let's use CPU in our example.
DEVICE_OPTION.device_type = caffe2_pb2.CPU

# If you have a GPU and want to run things there, uncomment the below two lines.
# If you have multiple GPUs, you also might want to specify a gpu id.
#DEVICE_OPTION.device_type = caffe2_pb2.CUDA
#DEVICE_OPTION.cuda_gpu_id = 0

# Caffe2 has a concept of "workspace", which is similar to that of Matlab. Each workspace
# is a self-contained set of tensors and networks. In this case, we will just use the default
# workspace, so we won't dive too deep into it.
workspace.SwitchWorkspace('default')

# First, we feed all the parameters to the workspace.
for param in tensors.protos:
    workspace.FeedBlob(param.name, param, DEVICE_OPTION)
# The network expects an input blob called "input", which we create here.
# The content of the input blob is going to be fed when we actually do
# classification.
workspace.CreateBlob("input")
# Specify the device option of the network, and then create it.
net.device_option.CopyFrom(DEVICE_OPTION)
workspace.CreateNet(net)


# location_prior defines the gaussian distribution for each location: it is a 3200x2
# matrix with the first dimension being the std and the second being the mean.
LOCATION_PRIOR = np.loadtxt(os.path.join(multibox_data_dir, 'ipriors800.txt'))

def RunMultiboxOnImage(image_file, location_prior):
    img = io.imread(image_file)
    resized_img = transform.resize(img, (224, 224))
    normalized_image = resized_img.reshape((1, 224, 224, 3)).astype(np.float32) - 0.5
    workspace.FeedBlob("input", normalized_image, DEVICE_OPTION)
    workspace.RunNet("multibox")
    location = workspace.FetchBlob("imagenet_location_projection").flatten(),
    # Recover the original locations
    location = location * location_prior[:,0] + location_prior[:,1]
    location = location.reshape((800, 4))
    confidence = workspace.FetchBlob("imagenet_confidence_projection").flatten()
    return location, confidence

def PrintBox(loc, height, width, style='r-'):
    """A utility function to help visualizing boxes."""
    xmin, ymin, xmax, ymax = loc[0] * width, loc[1] * height, loc[2] * width, loc[3] * height 
    #pyplot.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], style)
    return [xmin, ymin, xmax, ymax]

MOST_5_BOXES_ONLY=False
if MOST_5_BOXES_ONLY:
	my_dict_list=[]
	ppm_dir = '/proj/krishna/animation/images_downsample_2/'
	#'/proj/krishna/animation/face_detection/how_to_train_your_dragon_2/vad_motion_key_frames/'#'/proj/krishna/animation/HTD_scenes/shots_to_ppm/'
	all_conf=[]
	for idx,i in enumerate(os.listdir(ppm_dir)):
		my_dict={}
		img_name = os.path.join(ppm_dir,i)
		location, confidence = RunMultiboxOnImage(img_name, LOCATION_PRIOR)
		my_dict["image_path"] = img_name
		try: img = io.imread(img_name)
		except: IOError
		#pyplot.imshow(img)
		#pyplot.axis("off")
		# Let's show the most confident 5 predictions.
		# Note that argsort sorts things in increasing order.
		if not (idx%100): print img_name
		bboxes=[]
		conf=[]
		sorted_idx = np.argsort(confidence)
		p_conf = 1/(1+np.exp(-1*confidence))
		all_conf.append(p_conf)
		for idx in sorted_idx[-5:]:
		    bbox=PrintBox(location[idx], img.shape[0], img.shape[1])
		    bboxes.append(bbox)
		    conf.append(np.float(p_conf[idx]))
		    #print bboxes,
		#print ''
		my_dict["bbox"]=bboxes
		my_dict["conf"]=conf
		my_dict_list.append(my_dict)

	np.save('all_conf.npy', all_conf)

	with open('deep_multibox_HTD_downsample_frames.json','w') as outfile:
		json.dump(my_dict_list, outfile)
	print 'DONEEEEEEEEEEEEEEEEEEEEEEEE'
		#pyplot.show()


CONF_BOXES_ONLY=True
if CONF_BOXES_ONLY:

	my_dict_list=[]
	ppm_dir = '/proj/krishna/animation/images_downsample_2/'
	#'/proj/krishna/animation/face_detection/how_to_train_your_dragon_2/vad_motion_key_frames/'#'/proj/krishna/animation/HTD_scenes/shots_to_ppm/'
	all_conf=[]
	for idx,i in enumerate(os.listdir(ppm_dir)):
		my_dict={}
		img_name = os.path.join(ppm_dir,i)
		location, confidence = RunMultiboxOnImage(img_name, LOCATION_PRIOR)
		#my_dict["image_path"] = img_name
		#try: img = io.imread(img_name)
		#except: IOError
		#pyplot.imshow(img)
		#pyplot.axis("off")
		# Let's show the most confident 5 predictions.
		# Note that argsort sorts things in increasing order.
		if not (idx%100): print img_name
		bboxes=[]
		conf=[]
		sorted_idx = np.argsort(confidence)
		p_conf = 1/(1+np.exp(-1*confidence))
		conf_idx = np.where(p_conf>=0.1)
		#all_conf.append(p_conf)
		if len(conf_idx[0])>0:
			img = io.imread(img_name)
			my_dict["image_path"] = img_name
			for idx in conf_idx[0]:
			    bbox=PrintBox(location[idx], img.shape[0], img.shape[1])
			    bboxes.append([ bbox, np.float(p_conf[idx]) ])
			    #conf.append(np.float(p_conf[idx]))
		    #print bboxes,
		#print ''
			my_dict["bbox_conf"]=bboxes
			#my_dict["conf"]=conf
			my_dict_list.append(my_dict)

	#np.save('all_conf.npy', all_conf)

	with open('TMP_deep_multibox_HTD_downsample_frames_conf_only.json','w') as outfile:
		json.dump(my_dict_list, outfile)
	print 'DONEEEEEEEEEEEEEEEEEEEEEEEE'
		#pyplot.show()
