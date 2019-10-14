import cv2
import pandas as pd
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
    upper = np.array([int(colour+l*bounds),255,150])    #Currently lot of params are guesstimates, change them to make them statistical, such as 50% pixels h.
    mask = 255-cv2.inRange(img1, lower, upper)
    mask = cv2.bitwise_and(img1,img1, mask=mask)
    img1 = cv2.cvtColor(mask,cv2.COLOR_HSV2BGR)
    # cv2.imwrite(string,img1)
    return img1

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    try:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape((shape[0],shape[1]))
    except:
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        return img.reshape((shape[0],shape[1]))

# def get_seg_labels(labels,shape):
#   ret = np.zeros((shape[0],shape[1]))
#   for l in range(len(labels)):
#       ret += rle_decode(labels[l][1],shape) + l
#   return ret

if __name__ == "__main__":
    batch_size = 100#int(sys.argv[1])
    point_num = 50000#int(sys.argv[2])
    folder = './test_images'
    label_list = []
    label_file = './train.csv'
    data = np.zeros((batch_size, point_num, 6))
    final_label = np.zeros((batch_size,point_num), dtype=np.int32)
    point_num_total = 0 
    idx_img = 0
    labels = pd.read_csv(label_file)
    i = 0
    idx_h5 = 0
    while i < len(labels) // 4:
        filename = labels['Image_Label'][4*i].split('_')[0]
        # print(filename,labels['EncodedPixels'][4*i+0])
        image = cv2.imread(os.path.join(folder, filename))
        im_shape = image.shape
        im_labels = np.zeros((im_shape[0],im_shape[1]))
        for j in range(4):
            print(j,type(labels['EncodedPixels'][4*i+j]))
            l = labels['EncodedPixels'][4*i+j]
            im_labels = np.maximum(im_labels,(j+1)*rle_decode(l,im_shape))  #TODO Handle multiple labels
        new_image = get_mask_hue(image,k=5)
        points = np.nonzero(new_image.sum(axis=2))
        pixels = new_image[[points[0],points[1]]].reshape((len(points[0]),3))
        label_pixels = im_labels[[points[0],points[1]]].reshape((len(points[0]))) 
        points = np.concatenate([points[0].reshape((len(points[0]),1)),np.random.random((len(points[0]),1)) * 1e-6,points[1].reshape((len(points[0]),1))],axis=1)
        point_num_total = point_num_total + len(points)
        pixels_sum = np.sum(np.array(pixels))
        probs = (np.array(pixels)/pixels_sum).sum(axis=1)
        indices = np.random.choice(list(range(len(points))), size=point_num,
                                   replace=(len(points) < point_num), p=probs)
        pixels_array = (np.array(pixels)[indices].astype(np.float32) / 255) - 0.5
        points_array = np.array(points)[indices]
        label_array = np.array(label_pixels)[indices]
        idx_in_batch = idx_img % batch_size
        data[idx_in_batch, ...] = np.concatenate((points_array, pixels_array), axis=-1)
        final_label[idx_in_batch, ...] = label_array
        idx_img+=1
        print(idx_img%batch_size)
        if ((idx_img + 1) % batch_size == 0):# or idx_img == len(images) - 1:
            item_num = idx_in_batch + 1
            filename_h5 = os.path.join(os.path.dirname(folder), '%d.h5' % (idx_h5))
            print('Saving {}...'.format( filename_h5))
            # filelist_h5.write('./%d.h5\n' % (idx_h5))

            file = h5py.File(filename_h5, 'w')
            file.create_dataset('data', data=data[0:item_num, ...])
            file.create_dataset('label', data=final_label[0:item_num, ...])
            file.close()

            idx_h5 = idx_h5 + 1
        i+=4


    # for dirname, _, filenames in os.walk(folder):
    #   for filename in filenames:


                # print(j[1].shape)
                # get_mask_hue
        # i = cv2.imread('Data_4.jpg')
    # get_hist(i,'Hist2.jpg')
    # print(i[:,:,0].shape)
    # get_mask_hue(i,k=5,string='Data_4n3.jpg')