import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


if __name__ == "__main__":
    plt.ion()
    for j in range(10):
        samples = np.random.rand(16, 28, 28)

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis("off")
            plt.imshow(sample, cmap='Greys_r')
        plt.pause(1)
        plt.close(fig)




