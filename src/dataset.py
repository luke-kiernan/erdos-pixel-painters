from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torchvision.transforms as transforms

class BuildImageDataset(Dataset):
    def __init__(self, image_data, transform_in=None,transform_out=None):
        self.image_data = image_data
        self.transform_in = transform_in
        self.transform_out = transform_out

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        # This routine works for both CIFAR and COCO datasets
        img, _  = self.image_data[idx]
        if self.transform_in:
            input_image = self.transform_in(img)
        
        if self.transform_out:
            target_image = self.transform_out(img)
        
        return input_image, target_image
    
    def display_img_pair(self, idx, upscaling_factor=1):

        img, _  = self.image_data[idx]
        if self.transform_in:
            input_image = self.transform_in(img)
        #if self.transform_out:
        target_image = transforms.ToTensor()(img)
        plt.figure(figsize=(upscaling_factor, upscaling_factor))
        plt.subplot(1, 2, 1)
        if input_image.shape[0] == 1:
            plt.imshow(input_image[0], cmap='gray')
        else:
            warnings.warn("The input image has more than 1 channel. Displaying the first channel only.")
            plt.imshow(input_image[0], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        if target_image.shape[0] == 3:
            plt.imshow(np.transpose(target_image,(1,2,0)), interpolation='nearest')
        else:
            warnings.warn("The target image does not have 3 channels. Displaying the first channel only.")
            plt.imshow(img[0], cmap='gray')
        plt.axis('off')
        plt.show()