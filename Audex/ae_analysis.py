#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pt

from ae       import Autoencoder
from ae_train import load_mnist

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images       = images[sample_images_index]
    sample_labels       = labels[sample_images_index]
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images):
    fig = pt.figure(figsize=(15, 3))

    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    pt.show()

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    pt.figure(figsize=(10, 10))
    pt.scatter(latent_representations[:, 0],
               latent_representations[:, 1],
               cmap="rainbow", c=sample_labels, alpha=0.5, s=2)
    pt.colorbar()
    pt.show()

if __name__ == "__main__":
    ae = Autoencoder.load()
    x_train, y_train, x_test, y_test = load_mnist()

    num_reconstructed_images_to_show = 8
    sample_images, _        = select_images(x_test, y_test, num_reconstructed_images_to_show)
    reconstructed_images, _ = ae.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    num_latent_points_to_show = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_latent_points_to_show)
    _, latent_representations = ae.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)