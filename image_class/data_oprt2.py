import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from sklearn import preprocessing

"""
针对的是22导联、500Hz的脑电图，以20s为一个时间窗口，即10000行数据

"""

__PATH__ = "D:/data/XMU_EEG"
__FIG_PATH__ = "D:/train/"

#####                              #####
#            Data Processing           #
#####                              #####
#pandas 读取数据
def load_txt(filename,withHeader=False,withLabel=False,isArray=True,isDual=True,elec_num=22):
	# get label
	if withLabel:
		label = np.array([filename.split('/')[-1].split('_')[0]])
	# read file
	if(filename.split('.')[-1]!='txt'):
		filename = filename + '.txt'
	file = open(filename)
	lines = file.readlines()
	# slice data
	if isDual:
		offset = 1
	else:
		offset = 5
	if elec_num is not None:
		offset = -elec_num
	head = lines[0].split('\t')[:-offset]
	lines = lines[1:]
	data = []
	if isArray:
		for line in lines:
			line = line.strip('\n').split('\t')[:-offset]
			while '' in line:
				line.remove('')
			line = [float(d) for d in line]
			data.append(line)
		data = np.array(data)
	else:
		for line in lines:
			line = line.strip('\n').split('\t')[:-offset]
			while '' in line:
				line.remove('')
			line = [[float(d)] for d in line]
			data.append(line)
	# output
	output = []
	if withHeader:
		output.append(head)
	output.append(data)
	if withLabel:
		output.append(label)
	return output

def data_to_img(data,img_size=[256,256],offset_axis="x",norm=None,enhancement=True):
	xlist = data
	xrange = np.ceil(np.max(xlist)-np.min(xlist))
	yrange = len(xlist)
	ylist = np.array(range(yrange))
	img = np.zeros(img_size)
	# center the fig
	if offset_axis=="all":
		offset_x = img_size[0]/2
		offset_y = img_size[1]/2
	elif offset_axis=="x":
		offset_x = img_size[0]/2
		offset_y = 0
	elif offset_axis=="y":
		offset_x = 0
		offset_y = img_size[1]/2
	else:
		offset_x = 0
		offset_y = 0
	
	for (a,b) in zip(xlist,ylist):
		x = int(np.floor(a*img_size[0]/xrange)+offset_x)
		y = int(np.floor(b*img_size[1]/yrange)+offset_y)
		if x>(img_size[0]-1): x = img_size[0]-1
		if x<0: x = 0
		if enhancement:
			img[x][y]=255
		else:
			img[x][y]+=1
	return img

def img_to_data(filename,toMatrix=False):
	img = Image.open(filename)
	#img.show()
	width,height = img.size # width,height
	# remove color
	img = img.convert("L")
	data = img.getdata()
	if toMatrix:
		data = np.matrix(data,dtype='float')/255.0
	else:
		data = np.array(data,dtype='float')/255.0
	data = np.reshape(data,(height,width)) # height,width
	return data

def one_hot_encode(data,n_classes=1,isInt=False):
	if isInt:
		return np.eye(n_classes)[data]
	else:
		enc = preprocessing.OneHotEncoder()
		enc.fit(data)
		return enc.transform(data).toarray()
		
def data_slice(data,length,label=None,crop=False):
	data_len = len(data)
	data_slices = []
	# crop data
	n = data_len//length
	if not crop:
		n += 1
	if label is None:
		for i in range(n):
			data_slices.append(data[:length])
			data = data[length:]	
	else:
		for i in range(n):
			data_slices.append([data[:length],label])
			data = data[length:]
	return data_slices

def data_stretch(data,length,times=None,elec_num=22):
	new_img = []
	if times is None:
		times = length//elec_num
	for i in range(length):
		temp = []
		for d in data[i]:
			d = str(d)
			for _ in range(times):
				temp.append(float(d))
		new_img.append(temp)
	new_img = np.array(new_img)
	print(np.shape(new_img))
	return new_img
	
def get_dataset(path="/mnt/e/XMU_EEG/",length=10000,electrode_num=1):
	labeled_data = []
	if electrode_num<0:
		raise ValueError
	# get file name list
	filenamelist = glob.glob(os.path.join(path,"*.txt"))
	# get data from files
	for filename in filenamelist:
		print("Reading " + filename + "...")
		data_collection = load_txt(filename,isArray=True,withLabel=True)
		data = data_collection[0]
		label = data_collection[1]
		data_slices = data_slice(data,length=length,label=label,crop=True)
		if electrode_num>len(data[0]):
			raise ValueError
		for ds in data_slices:
			# select EEG data without label, .T:(N,22)->(22,N)
			ds = ds[0].T 
			if electrode_num==0:
				for i in range(len(ds)):
					labeled_data.append([ds[i],label])
			else:
				# select one electrode
				labeled_data.append([ds[electrode_num-1],label])
	print("Total number of files: %d" %len(filenamelist))
	print("Total number of data: %d" %len(labeled_data))
	#print("First data in dataset: {data}".format(data=labeled_data[0]))
	#print("First data in first data of dataset: {data}".format(data=labeled_data[0][0]))
	#print("Shape of First data in first data of dataset: {shape}" \
	#					.format(shape=np.shape(labeled_data[0][0])))
	return labeled_data
	
def split_train_test_data(x,y,test_size):
	#print(images)
	#print(labels)
	test_x = np.array(x[-test_size:])
	test_y = np.array(y[-test_size:])
	train_x = np.array(x[:-test_size])
	train_y = np.array(y[:-test_size])
	#print(len(images),len(images_test))
	#print("Images shape: {shape}".format(shape=np.shape(images)))
	#print(len(labels),len(test_y))
	#print("Labels shape: {shape}".format(shape=np.shape(labels)))
	return (train_x,train_y,test_x,test_y)

def shuffle_train_data(images,labels):
	data_size = len(images)
	perm = np.arange(data_size)
	np.random.shuffle(perm)
	images = np.array(images)[perm]
	labels = np.array(labels)[perm]
	return (images,labels)
	
def train_next_batch(step,images,labels,batch_size,isRecurrent=False):
	data_size = len(images)
	"""
	start = index_epoch
	index_epoch += batch_size
	if index_epoch > data_size:
		perm = np.arange(data_size)
		np.random.shuffle(perm)
		images = images[perm]
		labels = labels[perm]
		start = 0
		index_epoch = batch_size
	end = index_epoch
	"""
	if isRecurrent:
		start = (step*batch_size) % data_size
		if start + batch_size > data_size:
			end = start+batch_size-data_size
			images_ = np.r_[images[start:],images[:end]]
			labels_ = np.r_[labels[start:],labels[:end]]
		else:
			end = start+batch_size
			images_ = images[start:end]
			labels_ = labels[start:end]
		return (np.array(images_),np.array(labels_))
	else:
		start = (step*batch_size) % data_size
		end = min(start + batch_size, data_size)
		return (np.array(images[start:end]),np.array(labels[start:end]))
	
#####                              #####
#                Display               #
#####                              #####

def show_matrix_img(data):
	data = data*255
	img = Image.fromarray(data.astype(np.uint8))
	#img.show()
	#plt.show()
	img.save('matrix_img.png')
	
def show_plot(data,size=[256,256],dpi=64,linewidth=1,axis=True,setTicks=False):
	if (size and dpi) is not None:
		plt.figure(figsize=(size[0]//dpi,size[1]//dpi),dpi=dpi)
	plt.plot(data,linewidth=linewidth)
	# set range for axis
	# plt.xlim((a,b))
	# plt.ylim((c,d))
	# plt.xlabel('') # plt.ylabel('')
	# set scale for axis
	# x_ticks = np.arange(0,100,1)
	# y_ticks = np.arange(-1,1,0.2)
	# plt.xticks(np.arange(0,100,1))
	if setTicks:
		plt.yticks(np.arange(0,2,0.1)) #
	if not axis:
		plt.axis('off')
	plt.xlim(0,len(data))
	plt.show()

def show_subplot(data,num=0,header=None,size=[800,800],dpi=80,linewidth=1,axis=True,setTicks=False):
	if (size and dpi) is not None:
		fig = plt.figure(figsize=(size[0]//dpi,size[1]//dpi),dpi=dpi)
	else:
		fig = plt.figure()
	if num==0:
		num = len(data)
	xlim_len = len(data[0])
	for i in range(num):
		if header is not None:
			ax = fig.add_subplot(22,1,(i+1),ylabel=header[i])
		else:
			ax = fig.add_subplot(22,1,(i+1))
		ax.plot(data[i],linewidth=linewidth)
		if setTicks:
			ax.set_xticks(np.arange(0,xlim_len,1))
			ax.set_yticks(np.arange(0,2,0.5)) #
			#ax.set_ylim([0,2])
		if not axis:
			ax.axis('off')
	plt.show()

def show_overlap_plot(data,num=0,header=None,size=[800,800],dpi=80,linewidth=1,axis=True,setTicks=False):
	if num==0:
		num = len(data)
	xlim_len = len(data[0])
	if header is None:
		if (size and dpi) is not None:
			plt.figure(figsize=(size[0]//dpi,size[1]//dpi),dpi=dpi)
		#for i in range(num):
		#	plt.plot(data[i],linewidth=linewidth)
		plt.plot(data,linewidth=linewidth)
		plt.yticks(np.arange(0,2,0.1)) #
	else:
		if (size and dpi) is not None:
			ax = plt.figure(figsize=(size[0]//dpi,size[1]//dpi),dpi=dpi)
		else:
			ax = plt.figure()
		ax = plt.subplot(1,1,1)
		if setTicks:
			ax.set_xticks(np.arange(0,xlim_len,1))
			ax.set_yticks(np.arange(0,2,0.1)) #
		axn = [0]*len(data)
		for i in range(num):
			axn[i] = ax.plot(data[i],linewidth=linewidth,label=header[i])
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles[::-1],labels[::-1])
	if not axis:
		ax.axis('off')
	plt.tight_layout()
	plt.show()

def show_data_img(data,img_size=[256,256],offset_axis="x",norm=None,enhancement=True,axis=True):
	img = data_to_img(data,img_size=img_size,offset_axis=offset_axis,norm=norm,enhancement=enhancement)
	plt.imshow(img,cmap='gray')
	if not axis:
		plt.axis('off')
	plt.tight_layout()
	plt.show()
	
#####                              #####
#               Save fig               #
#####                              #####

def save_plot(filename,column=0,size=[256,256],dpi=64,linewidth=1,axis=True):
	data = load_txt(filename,withHeader=True,isArray=True)
	header = data[0]
	if column>len(header) or column<0:
		raise ValueError
	data = data[1].T
	if (size and dpi) is not None:
		plt.figure(figsize=(size[0]//dpi,size[1]//dpi),dpi=dpi)
	plt.plot(data[column],linewidth=linewidth)
	plt.xlim(0,len(data[column]))
	if not axis:
		plt.axis('off')
	figname = filename + '_' + header[column] + '.png'
	plt.tight_layout()
	plt.savefig(figname,bbox_inches='tight',pad_inches=0)

def save_many_txt_img(path="",size=[256,256],dpi=64,linewidth=1,axis=True):
	if not os.path.exists(path):
		print("making directory: %s" %path)
		os.makedirs(path)
	filenamelist = glob.glob(os.path.join(path,"*.txt"))
	for filename in filenamelist:
		data = load_txt(filename,withHeader=True,isArray=True)
		header = data[0]
		data = data[1].T
		# directory
		folder = "_".join(filename.split('_')[:-2])
		save_path = os.path.join(path,folder)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		print("Reading " + filename + "...")
		for i in range(len(data)):
			# save fig
			if (size and dpi) is not None:
				plt.figure(figsize=(size[0]//dpi,size[1]//dpi),dpi=dpi)
			plt.plot(data[i],linewidth=linewidth)
			plt.xlim(0,len(data[i]))
			if not axis:
				plt.axis('off')
			figname = header[i]+'.png'
			plt.savefig(os.path.join(save_path,figname),bbox_inches='tight',pad_inches=0)
			plt.close()

def save_subplotfig(data,figname,num=0,header=None,size=[800,800],dpi=80,linewidth=1,axis=True,setTicks=False):
	if (size and dpi) is not None:
		fig = plt.figure(figsize=(size[0]//dpi,size[1]//dpi),dpi=dpi)
	else:
		fig = plt.figure()
	if num==0:
		num = len(data)
	xlim_len = len(data[0])
	for i in range(num):
		if header is not None:
			ax = fig.add_subplot(22,1,(i+1),ylabel=header[i])
		else:
			ax = fig.add_subplot(22,1,(i+1))
		ax.plot(data[i],linewidth=linewidth)
		if setTicks:
			ax.set_xlim(0,xlim_len)
			ax.set_xticks(np.arange(0,xlim_len,1))
			ax.set_yticks(np.arange(0,2,0.5)) #
			#ax.set_ylim([0,2])
		if not axis:
			ax.axis('off')
	plt.tight_layout()
	plt.savefig(figname,bbox_inches='tight',pad_inches=0)

#####                              #####
#               Instance               #
#####                              #####

## tested
## Decrepit
def instance_get_dataset_from_seg():
	# constant
	path = "/home/acafl/eeg/data/XMU_EEG/"
	labeled_data = []
	all_electrode = False
	# get file name list
	filenamelist = glob.glob(os.path.join(path,"*.txt"))
	# get data from files
	for filename in filenamelist:
		print("Reading " + filename + "...")
		data_collection = load_txt(filename,isArray=True,withLabel=True)
		data = data_collection[0].T
		label = data_collection[1]
		if all_electrode:
		# select all electrode
			for i in range(len(data)):
				labeled_data.append([data[i],label])
		else:
		# select one electrode
			labeled_data.append([data[0],label])
	print("Total number of files: %d" %len(filenamelist))
	print("Total number of data: %d" %len(labeled_data))
	#print("First data in dataset: {data}".format(data=labeled_data[0]))
	#print("First data in first data of dataset: {data}".format(data=labeled_data[0][0]))
	#print("Shape of First data in first data of dataset: {shape}" \
	#					.format(shape=np.shape(labeled_data[0][0])))
	return labeled_data
	### for example
	"""
	filename = filenamelist[0]
	data_collection = load_txt(filename,isArray=True,withLabel=True)
	data = data_collection[0].T
	label = data_collection[1]
	print(data_collection)
	print(np.shape(data))
	print(data)
	print(label)
	"""

## tested
def instance_get_dataset_from_raw():
	# constant
	path = "/home/acafl/eeg/data/XMU_EEG/"
	length = 10000
	#get_dataset(path="/mnt/e/XMU_EEG/",length=10000,electrode_num=1)
	labeled_data = []
	electrode_num = 1
	# get file name list
	filenamelist = glob.glob(os.path.join(path,"*.txt"))
	# get data from files
	for filename in filenamelist:
		print("Reading " + filename + "...")
		data_collection = load_txt(filename,isArray=True,withLabel=True)
		data = data_collection[0]
		label = data_collection[1]
		data_slices = data_slice(data,length=length,label=label,crop=True)
		for ds in data_slices:
			# only select the data
			ds = ds[electrode_num-1].T
			if electrode_num==0:
				for i in range(len(ds)):
					labeled_data.append([ds[i],label])
			else:
				# select one electrode
				labeled_data.append([ds[0],label])
	print("Total number of files: %d" %len(filenamelist))
	print("Total number of data: %d" %len(labeled_data))
	#print("First data in dataset: {data}".format(data=labeled_data[0]))
	#print("First data in first data of dataset: {data}".format(data=labeled_data[0][0]))
	#print("Shape of First data in first data of dataset: {shape}" \
	#					.format(shape=np.shape(labeled_data[0][0])))
	return labeled_data
	"""
	### for example
	filename = filenamelist[10]
	data_collection = load_txt(filename,isArray=True,withLabel=True)
	data = data_collection[0]
	label = data_collection[1]
	# put in data of shape like (17000,22)
	data_slices = data_slice(data,length=length,label=label,crop=True)
	print(data_slices)
	print(np.shape(data_slices))
	for ds in data_slices:
			# only select the data
			ds = ds[0].T
			if all_electrode:
				for i in range(len(ds)):
					labeled_data.append([ds[i],label])
			else:
				# select one electrode
				labeled_data.append([ds[0],label])
	print(labeled_data[0])
	print(np.shape(labeled_data[0]))
	print(labeled_data[0][0])
	print(np.shape(labeled_data[0][0]))
	"""
	
## tested
def instance_get_separate_dataset(isImage = True):
	# constant
	path = "/home/acafl/eeg/data/XMU_EEG/"
	length = 10000
	# read data
	data = get_dataset(path=path,length=length,electrode_num=22)
	# shuffle
	random.shuffle(data)
	# separation
	array_data = [d[0] for d in data]
	print("EEG data shape: {shape}".format(shape=np.shape(array_data)))
	label_data = []
	for d in data:
		label = d[1][0]
		# 2 classes of labels: ictal and preictal
		if label == 'ictal':
			label_data.append(np.zeros([1]))
		elif label == 'preictal':
			label_data.append(np.zeros([1])+1)
	label_data = np.array(label_data).reshape(-1,1)
	print("Label data shape: {shape}".format(shape=np.shape(label_data)))
	#print("First 5 EEG data: {data}".format(data=array_data[:5]))
	#print("First 5 Label data: {data}".format(data=label_data[:5]))
	labels = one_hot_encode(label_data)
	#print("One hot encoding labels: {data}".format(data=labels[:5]))
	if not isImage:
		return (array_data,labels)
	else:
		images = []
		# array to image
		for a in array_data:
			img = data_to_img(a,img_size=[256,256],offset_axis="x",norm=None,enhancement=True)
			images.append(img)
		return (images,labels)

def instance_get_pro_separate_dataset(isImage = True):
	# constant
	path = "/home/acafl/eeg/data/XMU_EEG_PRO/"
	length = 5000
	# read data
	data = get_dataset(path=path,length=length,electrode_num=0)
	# shuffle
	random.shuffle(data)
	# separation
	array_data = [d[0] for d in data]
	print("EEG data shape: {shape}".format(shape=np.shape(array_data)))
	label_data = []
	for d in data:
		label = d[1][0]
		# 2 classes of labels: ictal and preictal
		if label == 'ictal':
			label_data.append(np.zeros([1]))
		elif label == 'preictal':
			label_data.append(np.zeros([1])+1)
		elif label == 'interictal':
			label_data.append(np.zeros([1])+2)
	label_data = np.array(label_data).reshape(-1,1)
	print("Label data shape: {shape}".format(shape=np.shape(label_data)))
	#print("First 5 EEG data: {data}".format(data=array_data[:5]))
	#print("First 5 Label data: {data}".format(data=label_data[:5]))
	labels = one_hot_encode(label_data)
	#print("One hot encoding labels: {data}".format(data=labels[:5]))
	if not isImage:
		return (array_data,labels)
	else:
		images = []
		# array to image
		for a in array_data:
			img = data_to_img(a,img_size=[256,256],offset_axis="x",norm=None,enhancement=True)
			images.append(img)
		return (images,labels)

def instance_get_separate_imgdataset():
	# constant
	path = __FIG_PATH__
	length = 5000
	# read data
	filenamelist = glob.glob(path+"*.png")
	#filename = filenamelist[0]
	img_data = []
	label_data = []
	for filename in filenamelist:
		#print("Reading " + filename + "...")
		img = img_to_data(filename,toMatrix=False)
		img_data.append(img)
		label = filename.split('\\')[-1].split('_')[0]
		#print(label)
		if label == 'ictal':
			label_data.append(np.zeros([1]))
		elif label == 'preictal':
			label_data.append(np.zeros([1])+1)
	# shuffle
	print(img_data,label_data)
	print(label_data )
	data = list(zip(img_data,label_data))
	random.shuffle(data)
	images,labels = zip(*data)
	# 2 classes of labels: ictal and preictal
	labels = np.array(labels).reshape(-1,1)
	labels = one_hot_encode(labels)
	return (images,labels)
		
## Tested
def instance_data_slice():
	path = "/home/acafl/eeg/data/XMU_EEG/"
	filenamelist = glob.glob(path+"*.txt")
	filename = filenamelist[10]
	length = 10000
	data_collection = load_txt(filename,withLabel=True)
	data = data_collection[0]
	print("Shape of this eeg data: {shape}".format(shape=np.shape(data)))
	label = data_collection[1]
	data_slices = data_slice(data,length=length,label=label,crop=True)
	print("Number of data slices: %d" %len(data_slices))
	if len(data_slices)==0:
		exit()
	print("First data slice: {slice}".format(slice=data_slices[0]))
	print("First data in slice: {data}".format(data=data_slices[0][0]))
	print("First label in slice: {label}".format(label=data_slices[0][1]))
	
def instance_save_all_img():
	path = '/mnt/e/XMU_EEG/'
	save_many_txt_img(path,axis=False)

def instance_show_subplot():
	path = ""
	filenamelist = glob.glob(path+"*.txt")
	filename = filenamelist[0]
	data = load_txt(filename,withHeader=True,isArray=True)
	header = data[0]
	data = data[1].T
	show_subplot(data,header=header,axis=True)
	
def instance_show_overlap_plot():
	path = ""
	filenamelist = glob.glob(path+"*.txt")
	filename = filenamelist[0]
	data = load_txt(filename,withHeader=True,isArray=True)
	header = data[0]
	data = data[1].T
	show_overlap_plot(data,axis=True,free=True)

def instance_show_array_img():
	data = load_txt('preictal_34-1_1_seg_1').T[0]
	img = data_to_img(data,img_size=[256,256])
	plt.imshow(img,cmap='gray')
	plt.show()

def instance_load_train_test_data():
	images,labels = instance_get_separate_dataset()
	#print(images)
	#print(labels)
	test_size = 29
	test_x = images[-test_size:]
	test_y = labels[-test_size:]
	train_x = images[:-test_size]
	train_y = labels[:-test_size]
	#print(len(images),len(test_x))
	print("Images shape: {shape}".format(shape=np.shape(images)))
	print(len(labels),len(test_y))
	print("Labels shape: {shape}".format(shape=np.shape(labels)))
	return (train_x,train_y,test_x,test_y)
	
if __name__ == '__main__':
	######        instance        ######
	######   with random number   ######
	#---- get filename ----#
	path="/home/acafl/eeg/data/XMU_EEG/"
	filenamelist = glob.glob(os.path.join(path,"*.txt"))
	#num = random.randrange(0,len(filenamelist),1)
	#filename = filenamelist[num]	
	length=10000
	#---- load data ----#
	filename = os.path.join(path,'ictal_35-1_1')
	data_collection = load_txt(filename,withHeader=True,withLabel=True)
	header = data_collection[0]
	data = data_collection[1].T # .T
	label = data_collection[2]
	#data_slices = data_slice(data,length=length,label=label,crop=True)
	#---- operation ----#
	#print(data)
	#print(np.shape(data))
	#print(label)
	#show_data_img(data[0],img_size=[4092,4092])
	#show_plot(data[0])
	#show_overlap_plot(data,header=header)
	#instance_get_separate_dataset()
	#data = img_to_data('ictalwith22.png',toMatrix=True)
	#print(data)
	#show_matrix_img(data)
	#show_subplot(data,axis=False)
	data,labels=instance_get_separate_imgdataset()
	images = [img.reshape([256*256]) for img in data]
	#split train and test data
	train_x,train_y,test_x,test_y = split_train_test_data(images,labels,120)
	print(np.shape(train_x),np.shape(train_y))
	data = list(zip(train_x,train_y))
	random.shuffle(data)
	train_x,train_y = zip(*data)
	b_x, b_y = train_next_batch(0,train_x,train_y,50)
	print(type(b_x),np.shape(b_x),type(b_y),np.shape(b_y))
	b_x = np.reshape(b_x,[-1,256,256,1])
	print(np.shape(b_x))
	print(b_x)
	
