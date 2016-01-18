'''
Created on Dec 14, 2015

@author: krsna
'''
import numpy as np
import cv2, os, sys, json
#import cv2.cv as cv

from pylab import *


# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	#if boxes.dtype.kind == "i":
	#	boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 	
	# grab the coordinates of the bounding boxes
	x1 = np.array([i[0] for i in boxes])
	y1 = np.array([i[1] for i in boxes])
	x2 = np.array([i[2] for i in boxes])
	y2 = np.array([i[3] for i in boxes])
	
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		#print overlap, '---overlap---' 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return np.array(boxes)[pick]

color = np.random.randint(0,255,(100,3))
color = np.vstack(([0,255,0],color))


def rgb_equalize(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2HSV)
    hsv_img_equ = hsv_img.copy()
    v_img = hsv_img[:,:,-1]
    v_img_equ = cv2.equalizeHist(v_img)
    hsv_img_equ[:,:,-1] = v_img_equ
    rgb_img_equ = cv2.cvtColor(hsv_img_equ,cv2.COLOR_HSV2BGR)
    return rgb_img_equ


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return x1,y1,x2,y2


DISPLAY=False
SAVE_FACE=True


# The following display routine is only for old  style JSON meaning bbox and conf were serparate keys
BOX_5_ONLY=False
if BOX_5_ONLY:
	# - This is the JSON file that is output from the deep_multibox_detect.py code - there are two version one with keys\
	# - "bbox" and "conf" - the other version with just one key "bbox_conf" with [ [b,b,o,x] ,c ]
	face_dict_list = json.load(open(str(sys.argv[1]),'rU'))
	print 'press ESC key to go to the next image'
	
	for list_i, face_dict in enumerate(face_dict_list):
		img_path = face_dict["image_path"]
		img_name = os.path.basename(img_path).split('.')[0]
		CONF_THR = 0.1
		if True: #not(int(img_name)%5):
			if max(face_dict['conf']) >= CONF_THR:
				if not(list_i % 100): print list_i, ' of ', len(face_dict_list)
				
				print 'showing... ',img_path#," --- has ", face_dict["number_faces"], " faces"
				img = cv2.imread(img_path)
			    
				img_equ = rgb_equalize(img)
				#gray_im_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				#gray_im = cv2.equalizeHist(gray_im_)
				#img_height, img_width = gray_im.shape
				vis = img.copy()
				vis_nms = img.copy()
				if True:
					bbox_list = face_dict['bbox']
					
					bbox_for_nms = [bbox_list[b_i] for b_i in range(len(bbox_list)) if face_dict["conf"][b_i] >= CONF_THR]
					if True: #len(bbox_for_nms)>1:
						conf_avg_nms = np.mean([c_i for c_i in face_dict["conf"] if c_i >= CONF_THR])
						bbox_nms_list = list(non_max_suppression(bbox_for_nms,0.50))
						print bbox_nms_list
						for bbox_nms_i, bbox_nms in enumerate(bbox_nms_list):
							x1_nms,y1_nms,x2_nms,y2_nms = draw_rects(vis_nms, [bbox_nms], color[bbox_nms_i])
							if SAVE_FACE:
								cv2.imwrite(os.path.join('confident_objects',\
									'NMS_%s_conf_%s_%s.ppm' % ( img_name, str(int(conf_avg_nms*100)), str(bbox_nms_i) ) ), \
										img[y1_nms:y2_nms,x1_nms:x2_nms,:], [int(cv2.IMWRITE_PXM_BINARY),1])

						for bbox_i,bbox in enumerate(bbox_list):
							if face_dict["conf"][bbox_i] >= CONF_THR:
								#b_box = [int(b) for b in bbox]
								x1,y1,x2,y2 = draw_rects(vis, [bbox], color[bbox_i])
								conf = int(np.round(face_dict["conf"][bbox_i]*100))
								if SAVE_FACE:
									cv2.imwrite(os.path.join('confident_objects',\
										'%s_conf_%s_%s.ppm' % ( img_name, str(conf), str(bbox_i)) ), \
											img[y1:y2,x1:x2,:], [int(cv2.IMWRITE_PXM_BINARY),1])
							
						if DISPLAY:
							#cv2.imshow("faces",vis[y1:y2,x1:x2,:])
							cv2.imshow("faces",np.hstack((vis,vis_nms)))
							cv2.waitKey(0)
		
		
		

	cv2.destroyAllWindows()
	


DISPLAY = False
SAVE_FACE = True

# The following display routine is only for newer style JSON meaning bbox and conf were in the same list with key bbox_conf
BOX_CONF_ONLY=True
if BOX_CONF_ONLY:
	CONF_THR = 0.1
	objects_out_dir = '/proj/krishna/animation/multibox_nn/confident_objects/how_to_train_your_dragon_2/'
	face_dict_list = json.load(open(str(sys.argv[1]),'rU'))
	print 'press ESC key to go to the next image'
	
	for list_i, face_dict in enumerate(face_dict_list):
		img_path = face_dict["image_path"]
		img_name = os.path.basename(img_path).split('.')[0]
		
		bbox_list = [i[0] for i in face_dict['bbox_conf']]
		conf_list = [i[1] for i in face_dict['bbox_conf']]
		
		if True: #not(int(img_name)%5):
			if max(conf_list) >= CONF_THR:
				if not(list_i % 100): print list_i, ' of ', len(face_dict_list)
				
				print 'showing... ',img_path#," --- has ", face_dict["number_faces"], " faces"
				img = cv2.imread(img_path)
			    
				img_equ = rgb_equalize(img)
				#gray_im_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				#gray_im = cv2.equalizeHist(gray_im_)
				#img_height, img_width = gray_im.shape
				vis = img.copy()
				vis_nms = img.copy()
				if True:
					bbox_for_nms = [bbox_list[b_i] for b_i in range(len(bbox_list)) if conf_list[b_i] >= CONF_THR]
					if True: #len(bbox_for_nms)>1:
						conf_avg_nms = np.mean([c_i for c_i in conf_list if c_i >= CONF_THR])
						bbox_nms_list = list(non_max_suppression(bbox_for_nms,0.50))
						#print bbox_nms_list
						for bbox_nms_i, bbox_nms in enumerate(bbox_nms_list):
							x1_nms,y1_nms,x2_nms,y2_nms = draw_rects(vis_nms, [bbox_nms], color[bbox_nms_i])
							if SAVE_FACE:
								cv2.imwrite(os.path.join(objects_out_dir,\
									'NMS_%s_conf_%s_%s.ppm' % ( img_name, str(int(conf_avg_nms*100)), str(bbox_nms_i) ) ), \
										img[y1_nms:y2_nms,x1_nms:x2_nms,:], [int(cv2.IMWRITE_PXM_BINARY),1])

						for bbox_i,bbox in enumerate(bbox_list):
							if conf_list[bbox_i] >= CONF_THR:
								#b_box = [int(b) for b in bbox]
								x1,y1,x2,y2 = draw_rects(vis, [bbox], color[bbox_i])
								conf = int(np.round(conf_list[bbox_i]*100))
								if SAVE_FACE:
									cv2.imwrite(os.path.join(objects_out_dir,\
										'%s_conf_%s_%s.ppm' % ( img_name, str(conf), str(bbox_i)) ), \
											img[y1:y2,x1:x2,:], [int(cv2.IMWRITE_PXM_BINARY),1])
							
						if DISPLAY:
							#cv2.imshow("faces",vis[y1:y2,x1:x2,:])
							cv2.imshow("faces",np.hstack((vis,vis_nms)))
							cv2.waitKey(0)
		
	cv2.destroyAllWindows()
	







