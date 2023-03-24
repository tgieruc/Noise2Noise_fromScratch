from model import Model

import torch
import numpy as np
import torchvision.transforms as transforms

from matplotlib import pyplot as plt



class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.datasets = torch.cat([dataset1[:, None], dataset2[:, None]], dim=1)
        self.transforms = torch.nn.Sequential(
            # transforms.TrivialAugmentWide()
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.RandomCrop(size=(32, 32)),
            # transforms.ColorJitter(brightness=.5, hue=.3)
        )

    def __getitem__(self, i):
        if torch.rand(1) > 0.5:
            return self.transforms(self.datasets[i])
        else:
            return self.transforms(self.datasets[i, [1, 0]])

    def __len__(self):
        return len(self.datasets)

def display_rgb(img):
    plt.imshow(img.permute(1,2,0))
    plt.show()

def psnr_eval(model, noised, ground_truth, must_randomize=True):
    def psnr(denoised, ground_truth):
        mse = torch.mean((denoised.cpu() - ground_truth.cpu()) ** 2)
        return -10 * torch.log10(mse + 10 ** -8)
    clean_imgs = ground_truth.clone()
    noised = noised.float()
    ground_truth = ground_truth.float()
    denoised = noisy_imgs.clone()
    for i in range(len(noisy_imgs)):
        denoised[i] = model.predict(noisy_imgs[i].unsqueeze(0)) / 255


    psnr_result = psnr(denoised, (ground_truth / 255)).item()
    print(f'PSNR result: {psnr_result}dB')

    nb_images = 3

    f, axarr = plt.subplots(nb_images, 3)

    if must_randomize:
        nb_index = np.random.choice(len(noised), nb_images)
    else:
        nb_index = np.arange(nb_images)
    axarr[0, 0].set_title("Noisy Images")
    axarr[0, 1].set_title("Denoised")
    axarr[0, 2].set_title("Ground Truth")

    for i, index in enumerate(nb_index):
        axarr[i, 0].imshow(noised[index].permute(1,2,0).int())
        axarr[i,0].get_yaxis().set_visible(False)
        axarr[i,0].get_xaxis().set_visible(False)
        axarr[i, 1].imshow(denoised[index].cpu().detach().permute(1,2,0))
        axarr[i, 1].get_yaxis().set_visible(False)
        axarr[i, 1].get_xaxis().set_visible(False)
        axarr[i, 2].imshow(clean_imgs[index].permute(1,2,0))
        axarr[i, 2].get_yaxis().set_visible(False)
        axarr[i, 2].get_xaxis().set_visible(False)
    plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'The model will be loaded on the {"GPU" if device == "cuda" else "cpu"}.')

noisy_imgs_1, noisy_imgs_2 = torch.load('train_data.pkl')
noisy_imgs, clean_imgs = torch.load('val_data.pkl')


# Create a model
model = Model()
# train the model 
model.train(noisy_imgs_1, noisy_imgs_2, num_epochs=1)

#eval
psnr_eval(model, noisy_imgs, clean_imgs, must_randomize=False)


