import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import os
import h5py

def plot_image_rgb(img,graph):
	b,g,r = cv2.split(img)       # get b,g,r
	rgb_img = cv2.merge([r,g,b])     # switch it to rgb	
	graph.imshow(rgb_img)
def get_hist(img,name):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
	hist2 = cv2.calcHist([img],[1],None,[256],[0,256])
	hist3 = cv2.calcHist([img],[2],None,[256],[0,256])
	fig,ax = plt.subplots(4)
	ax[0].imshow(img)
	ax[1].plot(hist1[:])
	ax[2].plot(hist2[:])
	ax[3].plot(hist3[10:])

	plt.savefig(name)

def get_mask(img,t1=[60,20,5],t2=[50,10,5],t3=[40,10,5]):
	# print(np.max(img))
	img1 = img>180
	# img1 = np.expand_dims(((img[:,:,0]<t1[0]) * (img[:,:,0]>t1[1])),axis=2)
	# img2 = np.expand_dims(((img[:,:,1]<t2[0]) * (img[:,:,1]>t2[1])),axis=2)
	# img3 = np.expand_dims(((img[:,:,2]<t3[0]) * (img[:,:,2]>t3[1])),axis=2)	
	fig,ax = plt.subplots(2)
	# plot_image_rgb(np.concatenate([img1,img2,img3],axis=2).astype('f'),ax[1])
	plot_image_rgb(img1.astype('f'),ax[1])
	plot_image_rgb(img,ax[0])
	# ax[1].imshow(img1.astype('f'))
	# ax[0].imshow(img)
	plt.savefig('mask.jpg')

def get_mask_hue(img,k=5,l=3,string='NewMask1'):
	img1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	hist1 = cv2.calcHist([img1],[0],None,[180],[0,180]).reshape((180,))
	# print(hist1[5:].shape)
	top5 = hist1[5:].argsort()[::-1][:k]
	# hist1 = np.cumsum(hist1[5:][top5])/(np.sum(hist1))
	# print(hist1)
	colour = np.median(top5)
	bounds = max(colour-np.min(top5),np.max(top5)-colour)
	lower = np.array([int(colour-l*bounds),0,0])
	upper = np.array([int(colour+l*bounds),255,150])	#Currently lot of params are guesstimates, change them to make them statistical, such as 50% pixels h.
	mask = 255-cv2.inRange(img1, lower, upper)
	mask = cv2.bitwise_and(img1,img1, mask=mask)
	img1 = cv2.cvtColor(mask,cv2.COLOR_HSV2BGR)
	# cv2.imwrite(string,img1)
	return img1

# def probabilistic_dist_image(img,num_points):

if __name__ == "__main__":
	batch_size = 100#int(sys.argv[1])
	point_num = 5000#int(sys.argv[2])
	folder = './test_images'
	data = np.zeros((batch_size, point_num, 6))
	label = np.zeros((batch_size), dtype=np.int32)
	point_num_total = 0	
	idx_img = 0
	for dirname, _, filenames in os.walk(folder):
		idx_h5 = 0
		for filename in filenames:
			i = cv2.imread(os.path.join(dirname, filename))
			new_image = get_mask_hue(i,k=5)
			# print(new_image.shape)
			# points = []
			# pixels = []	
			# for x in range(new_image.shape[0]):
			# 	for y in range(new_image.shape[1]):
			# 		# print(new_image[x,y].shape)
			# 		if(new_image[x,y,0] + new_image[x,y,1] + new_image[x,y,2]==0):
			# 			continue
			# 		points.append((x, np.random.random() * 1e-6, y))
			# 		pixels.append(new_image[x,y])					
			points = np.nonzero(new_image.sum(axis=2))
			# print(points[0].shape)
			pixels = new_image[[points[0],points[1]]].reshape((len(points[0]),3))
			points = np.concatenate([points[0].reshape((len(points[0]),1)),np.random.random((len(points[0]),1)) * 1e-6,points[1].reshape((len(points[0]),1))],axis=1)
			# print(points.shape)
			point_num_total = point_num_total + len(points)
			pixels_sum = np.sum(np.array(pixels))
			probs = (np.array(pixels)/pixels_sum).sum(axis=1)
			# print(probs.shape)
			# probs = [pixel / pixels_sum for pixel in pixels]
			indices = np.random.choice(list(range(len(points))), size=point_num,
									   replace=(len(points) < point_num), p=probs)
			pixels_array = (np.array(pixels)[indices].astype(np.float32) / 255) - 0.5
			points_array = np.array(points)[indices]
			idx_in_batch = idx_img % batch_size
			data[idx_in_batch, ...] = np.concatenate((points_array, pixels_array), axis=-1)
			idx_img+=1
			print(idx_img%batch_size)
			if ((idx_img + 1) % batch_size == 0):# or idx_img == len(images) - 1:
				item_num = idx_in_batch + 1
				filename_h5 = os.path.join(os.path.dirname(folder), '%d.h5' % (idx_h5))
				print('Saving {}...'.format( filename_h5))
				# filelist_h5.write('./%d.h5\n' % (idx_h5))

				file = h5py.File(filename_h5, 'w')
				file.create_dataset('data', data=data[0:item_num, ...])
				file.create_dataset('label', data=label[0:item_num, ...])
				file.close()

				idx_h5 = idx_h5 + 1


				# print(j[1].shape)
		        # get_mask_hue
	# i = cv2.imread('Data_4.jpg')
	# get_hist(i,'Hist2.jpg')
	# print(i[:,:,0].shape)
	# get_mask_hue(i,k=5,string='Data_4n3.jpg')