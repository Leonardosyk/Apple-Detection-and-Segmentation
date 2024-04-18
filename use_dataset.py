import os
import torch
import matplotlib.pyplot as plt

# Make sure to define base_dir according to your file structure
base_dir = 'S:\\Mv_py_ass'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Define a function to load the dataset from a .pth file
def load_dataset(file_name):
    data = torch.load(file_name)
    images = data['images']
    labels = data['labels']
    return images, labels


# Load the datasets
train_images, train_labels = load_dataset(os.path.join(base_dir, 'train_dataset.pth'))
val_images, val_labels = load_dataset(os.path.join(base_dir, 'val_dataset.pth'))
test_images, test_labels = load_dataset(os.path.join(base_dir, 'test_dataset.pth'))


# Define a function to show images and labels
def show_images_and_labels(images, labels, title):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    fig.suptitle(title, fontsize=16)

    # Reverse normalization
    mean = 0.5
    std = 0.5
    for i in range(5):
        img = images[i].permute(1, 2, 0)  # Rearrange dimensions to fit the expected format
        img = img * std + mean  # Reverse normalization
        img = img.clamp(0, 1)  # Clamp values to the [0, 1] range
        label = labels[i].argmax(dim=0).item()

        axes[i].imshow(img.numpy(), cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()


# Display the first 5 images and labels from each dataset
show_images_and_labels(train_images, train_labels, 'Train Set')
show_images_and_labels(val_images, val_labels, 'Validation Set')
show_images_and_labels(test_images, test_labels, 'Test Set')
