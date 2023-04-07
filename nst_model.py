import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torchvision import datasets
from style_transfer_network import StyleTransferNet
from perceptual_loss_network import PerceptualLossNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler


class SequentialSubsetSampler(Sampler):
    r"""Samples elements sequentially, always in the same order from a subset defined by size.

    Arguments:
        data_source (Dataset): dataset to sample from
        subset_size: defines the subset from which to sample from
    """

    def __init__(self, data_source, subset_size):
        assert isinstance(
            data_source,
            Dataset) or isinstance(
            data_source,
            datasets.ImageFolder)
        self.data_source = data_source

        if subset_size is None:  # if None -> use the whole dataset
            subset_size = len(data_source)
        assert 0 < subset_size <= len(
            data_source), f'Subset size should be between (0, {len(data_source)}].'
        self.subset_size = subset_size

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size


class StyleTransferModel(object):
    def __init__(self, n_epoch, b_size):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_epoch = n_epoch
        self.batch_size = b_size

        self.content_losses = []
        self.style_losses = []
        self.tv_losses = []

        self.content_img_dir = os.path.join(
            os.getcwd(), 'images', 'content_image')
        self.style_img_dir = os.path.join(
            os.getcwd(), 'images', 'content_image', 'style_image')
        self.stylize_img_dir = os.path.join(
            os.getcwd(), 'images', 'content_image', 'stylize_image')

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        h, w, c = img.shape

        img = img.astype(np.float32)
        img /= 255.0
        img = torch.Tensor(img)
        img = img.view(-1, c, h, w)
        img = img.to(self.device)
        return img

    def save_image(self, img, img_path):
        img = img[:, :, ::-1]
        plt.imshow(img)
        plt.show()
        cv2.imwrite(img_path, img)

    def gram_matrix(self, A):
        return torch.matmul(torch.transpose(A, 2, 3), A)

    def get_training_data_loader(
            self,
            dataset_path,
            should_normalize=True,
            is_255_range=False):
        """
            There are multiple ways to make this feed-forward NST working,
            including using 0..255 range (without any normalization) images during transformer net training,
            keeping the options if somebody wants to play and get better results.
        """
        IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])
        IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])
        IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
        IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

        transform_list = [transforms.Resize(256),
                          transforms.CenterCrop(256),
                          transforms.ToTensor()]
        if is_255_range:
            transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
        if should_normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=IMAGENET_MEAN_255,
                    std=IMAGENET_STD_NEUTRAL) if is_255_range else transforms.Normalize(
                    mean=IMAGENET_MEAN_1,
                    std=IMAGENET_STD_1))
        transform = transforms.Compose(transform_list)

        train_dataset = datasets.ImageFolder(
            root=dataset_path, transform=transform)
        sampler = SequentialSubsetSampler(train_dataset, None)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True)
        return train_loader

    def total_variation(self, y):
        term1 = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
        term2 = torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
        return term1 + term2

    def training(
            self,
            alpha,
            beta,
            gamma,
            style_img_path,
            dataset_path,
            model_name):
        #        content_img = self.load_image(content_img_path)
        style_img = self.load_image(style_img_path)

        style_transfer_net = StyleTransferNet().to(self.device)
        perceptual_loss_net = PerceptualLossNet().to(self.device)

        optimizer = optim.Adam(style_transfer_net.parameters(), lr=0.001)

        style_img = self.load_image(style_img_path)
        phi_style = perceptual_loss_net(style_img)
        phi_style_gram = [self.gram_matrix(m) for m in phi_style.values()]

        train_loader = self.get_training_data_loader(dataset_path)
        n = len(train_loader.dataset) / self.batch_size
        for epoch in range(self.n_epoch):
            for i, (content_batch, _) in enumerate(train_loader):
                print(
                    'Epoch: %d/%d, Batch: %d/%d' %
                    (epoch + 1, self.n_epoch, i + 1, n))
                content_batch = content_batch.to(self.device)
                stylize_batch = style_transfer_net(content_batch)

                phi_content = perceptual_loss_net(content_batch)
                phi_stylize = perceptual_loss_net(stylize_batch)
                content_loss = nn.MSELoss('mean')(
                    phi_content['relu2_2'], phi_stylize['relu2_2'])

                style_loss = 0.0
                phi_stylize_gram = [
                    self.gram_matrix(m) for m in phi_stylize.values()]
                for y, y_s in zip(phi_style_gram, phi_stylize_gram):
                    style_loss += nn.MSELoss(reduction='mean')(y, y_s)
                style_loss /= len(phi_stylize_gram)

                tv_loss = self.total_variation(stylize_batch)

                total_loss = alpha * content_loss + beta * style_loss + gamma * tv_loss
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                self.content_losses.append(content_loss)
                self.style_losses.append(style_loss)
                self.tv_losses.append(tv_loss)

        torch.save(
            style_transfer_net.state_dict(),
            os.path.join(os.getcwd(), 'models', model_name)
        )

    def stylize(self, content_img_path, model_path):
        content_img = self.load_image(content_img_path)
        base = os.path.basename(content_img_path)
        base = os.path.splitext(base)[0]
        stylize_img_path = os.path.join(self.stylize_img_dir, base + '.png')

        st_net = StyleTransferNet()
        st_net.load_state_dict(torch.load(model_path))
        st_net.eval()

        with torch.no_grad():
            stylize_img = st_net(content_img).to('cpu').numpy()[0]
            self.save_image(stylize_img, stylize_img_path)


if __name__ == "__main__":
    model = StyleTransferModel(n_epoch=2, b_size=8)
    model.training(
        alpha=1, beta=400000, gamma=1,
        style_img_path='./images/style_image/Composition_VIII.jpg',
        dataset_path='./dataset/mscoco/',
        model_name='Composition.pt'
    )
    torch.cuda.empty_cache()
