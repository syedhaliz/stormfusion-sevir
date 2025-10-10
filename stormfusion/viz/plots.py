
import matplotlib.pyplot as plt
import numpy as np

def show_triplet(inp, truth, pred, t_index=0, savepath=None):
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(inp, cmap='magma'); axs[0].set_title('Input (t=12)'); axs[0].axis('off')
    axs[1].imshow(truth, cmap='magma'); axs[1].set_title('Truth (t=13)'); axs[1].axis('off')
    axs[2].imshow(pred, cmap='magma'); axs[2].set_title('Prediction (t=13)'); axs[2].axis('off')
    if savepath: fig.savefig(savepath, bbox_inches='tight')
    return fig
