import os
import struct
import numpy as np
import matplotlib.pyplot as plt


# load mnist data from path
def fun_load_mnist(path, kind="train"):
    labels_path = os.path.join(path, "%s-labels.idx1-ubyte" % kind)
    images_path = os.path.join(path, "%s-images.idx3-ubyte" % kind)
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, "rb") as impath:
        magic, num, rows, cols = struct.unpack(">IIII", impath.read(16))
        images = np.fromfile(impath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def next_batch(num, data, labels):
    # Return a total of `num` random samples and labels.
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


if __name__ == "__main__":
    train_images, train_labels = fun_load_mnist("../datasets", kind="train")
    test_images, test_labels = fun_load_mnist("../datasets", kind="t10k")
    print("train_images:", train_images[0], sep="\n")
    print("train_labels:", train_labels, sep="\n")
    print("test_images:", test_images, sep="\n")
    print("test_labels:", test_labels, sep="\n")

    '''
    # visualize
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(10):
        img = train_images[i].reshape(28, 28)
        ax[i].imshow(img, cmap="Greys", interpolation="nearest")    # plot
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax1 = ax1.flatten()
    for i in range(25):
        img = train_images[train_labels == 7][i].reshape(28, 28)
        ax1[i].imshow(img, cmap="Greys", interpolation="nearest")
    ax1[0].set_xticks([])
    ax1[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    '''





