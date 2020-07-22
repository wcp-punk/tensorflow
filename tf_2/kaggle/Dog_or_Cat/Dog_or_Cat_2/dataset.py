import os, glob
import random, csv
import tensorflow as tf

def load_csv(root, filename, name2label):
    
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        images += glob.glob(os.path.join(root, '*.jpg'))
        print(len(images), images)
        random.shuffle(images) # 随机打散顺序
        
        # 创建csv文件，并存储图片路径及其label信息
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-1].split('.')[0]
                label = name2label[name]
                writer.writerow([img, label])
            print('written into csv file:', filename)
        
    # 此时已经有csv文件，直接读取
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label)
            images.append(img)
            labels.append(label)
    # 返回图片路径list和标签list
    return images, labels


def load_dog_or_cat_data(root, mode='train'):
    # 创建数字编码表
    name2label = {'cat':0,'dog':1}
    
    images, labels = load_csv(root, 'images_train.csv', name2label)
    
    if mode == 'train':  # 80%
        images = images[:int(0.8 * len(images))]
        labels = labels[:int(0.8 * len(labels))]
    elif mode == 'val':  # 20%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    
    return images, labels, name2label

# 这里的mean和std根据真实的数据计算获得，比如ImageNet
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])
def normalize(x, mean=img_mean, std=img_std):
    # 标准化
    # x: [224, 224, 3]
    # mean: [224, 224, 3], std: [3]
    x = (x - mean)/std
    return x

def denormalize(x, mean=img_mean, std=img_std):
    # 标准化的逆过程
    x = x * std + mean
    return x

def preprocess(x,y):
    # x: 图片的路径List，y：图片的数字编码List
    x = tf.io.read_file(x) # 根据路径读取图片
    x = tf.image.decode_jpeg(x, channels=3) # 图片解码
    x = tf.image.resize(x, [128, 128]) # 图片缩放

    # 数据增强
    # x = tf.image.random_flip_up_down(x)
    x= tf.image.random_flip_left_right(x) # 左右镜像
    # 转换成张量
    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    x = normalize(x) # 标准化
    y = tf.convert_to_tensor(y) # 转换成张量

    return x, y

def main():
    import time
    batch_size = 15
    # 加载pokemon数据集，指定加载训练集
    images, labels, table = load_dog_or_cat_data(r'D:\data\kaggle\dogs-vs-cats\train', 'val')
    print('images:', len(images), images)
    print('labels:', len(labels), labels)
    print('table:', table)
    
    # images: string path
    # labels: number
    db = tf.data.Dataset.from_tensor_slices((images, labels))
    db = db.shuffle(1000).map(preprocess).batch(batch_size)
    
    
    # 创建TensorBoard对象
    writter = tf.summary.create_file_writer('logs')
    for step, (x,y) in enumerate(db):
        # x: [32, 224, 224, 3]
        # y: [32]
        with writter.as_default():
            x = denormalize(x) # 反向normalize，方便可视化
            # 写入图片数据
            tf.summary.image('img',x,step=step,max_outputs=9)
            time.sleep(5)
            
if __name__ == '__main__':
    main()
    























