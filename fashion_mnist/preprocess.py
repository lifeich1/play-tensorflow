import download as dl

train_images = dl.train_images / 255.0
test_images = dl.test_images / 255.0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10,))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(dl.class_names[dl.train_labels[i]])
    plt.show()
