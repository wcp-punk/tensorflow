import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return tf.image.flip_up_down(img)
    elif mode == 2:
        return tf.image.rot90(img)
    elif mode == 3:
        return tf.image.flip_up_down(tf.image.rot90(img))
    elif mode == 4:
        return tf.image.rot90(img, k=2)
    elif mode == 5:
        return tf.image.flip_up_down(tf.image.rot90(img, k=2))
    elif mode == 6:
        return tf.image.rot90(img, k=3)
    elif mode == 7:   
        return tf.image.flip_up_down(tf.image.rot90(img, k=3)) 


def gen_patches(file_name):

    # read image
    img = tf.io.read_file(file_name)  # 根据路径读取图片
    img = tf.image.decode_png(img, channels=0)  # 图片解码
    h, w, _ = img.shape
    patches = []

    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = tf.image.resize(img, [h_scaled, w_scaled], method='bicubic')
        img_scaled = tf.reshape(img_scaled, [h_scaled, w_scaled, 1])  # 图片缩放
        img_scaled = tf.cast(img_scaled, dtype=tf.uint8)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                #patches.append(x)        
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=tf.random.uniform(shape=[], minval=0, maxval=8, dtype=tf.int32))
                    patches.append(x_aug)

                
    return patches

def datagenerator(data_dir='data/Train400',verbose=False):
    
    file_list = tf.io.gfile.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        print(i)
        patch = gen_patches(file_list[i])
        if i == 0:
            data = patch
        else:
            data = tf.concat([data, patch], axis=0)  # 在科目维度上拼接
        # data.append(patch)
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done ^_^')
    
    data=data.numpy()
    writer = tf.io.TFRecordWriter("train400.tfrecords")
    for i in range(len(data)):
        x = data[i]  # 根据路径读取图片
        x = x.tobytes()
        x_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
        features = tf.train.Features(feature={"train400": x_feature})
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
        
    print('^_^-training data finished-^_^')

if __name__ == '__main__':   

    data = datagenerator(data_dir='data/Train400')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    