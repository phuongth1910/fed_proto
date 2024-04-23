from PIL import Image
import os.path
import torch
import warnings
import torch.utils.data as data
from torchvision import transforms
import numpy as np

def load_image_path(key, out_field, d):
    out_field = Image.open(d).convert('L')
    return out_field

def convert_tensor(key, d):
    # d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    c=torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    d=(255.0-c)/255.0
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

def convert_dict(k, v):
    return { k: v }

training_file = 'training.pt'
test_file = 'test.pt'
classes = ['0 - Not dead', '1 - Deadth']

class MRI_BRAIN(data.Dataset):
    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
            self.root = os.path.expanduser(root)
            self.transform = transform
            self.target_transform = target_transform
            self.train = train  # training set or test set

            # s_list = random.sample(range(0, 7), num_users)
            if self.train:
                # data_file = self.training_file
                self.data, self.targets = self.generate_ds(args, self.root)
                # self.loader = self.generate_ds(args, self.root)
            else:
                # data_file = self.test_file
                self.data, self.targets = self.generate_ds_test(args, self.root)
                # self.loader = self.generate_ds_test(args, self.root)
            # self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.open(img).convert('L')
            # loader = transforms.Compose([transforms.ToTensor()])
            # img = loader(img).unsqueeze(0)[0, 0, :, :]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

    def __len__(self):
            return len(self.data)
    
    # def generate_ds(self, args, root):
    #     # read 100 images per class per style
    #     num_class = args.num_classes
    #     num_img = args.train_shots_max * args.num_users

    #     data = []
    #     targets = torch.zeros([num_class  * num_img])
    #     files = os.listdir(os.path.join(root, 'data', 'raw_data', 'by_class'))

    #     for i in range(num_class):
    #         for k in range(num_img):
    #             img = os.path.join(root, 'data', 'raw_data', 'by_class', files[i], 'train_' + files[i], 'train_' + files[i] + '_'+str("%05d"%k)+'.png')
    #             data.append(img)
    #             targets[i * num_img + k] = i

    #     targets = targets.reshape([num_class * num_img])

    #     return data, targets
    
    def generate_ds(self, args, root):
        num_class = args.num_classes
        num_img = args.train_shots_max * args.num_users

        data = []
        targets = torch.zeros([num_class * num_img])

        for data_root in os.listdir(root):
            class_path = os.path.join(root, data_root)
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                for i in range(num_class):
                    for k in range(num_img):
                        img = os.path.join(class_path, files[i], 'train_' + files[i], 'train_' + files[i] + '_'+str("%05d"%k)+'.tif')
                        data.append(img)
                        targets[i * num_img + k] = i

        targets = targets.reshape([num_class * num_img])


        return data, targets

    def generate_ds_test(self, args, root):
        num_class = args.num_classes
        num_img = args.train_shots_max * args.num_users

        data = []
        targets = torch.zeros([num_class * num_img])

        for data_root in os.listdir(root):
            class_path = os.path.join(root, data_root)
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                # print(files)
                for i in range(num_class):
                    for k in range(num_img):
                        img = os.path.join(class_path, files[i], 'train_' + files[i], 'train_' + files[i] + '_'+str("%05d"%k)+'.tif')
                        data.append(img)
                        targets[i * num_img + k] = i

        targets = targets.reshape([num_class * num_img])


        return data, targets