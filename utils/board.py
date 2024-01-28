import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def get_output_figure(log):
    challenge = log["challenge"]
    real_response = log["real_response"]
    gen_response = log["gen_response"]

    cols = 4
    rows = 5
    fig, axs = plt.subplots(rows, cols, figsize=(16, 16))
    for row in range(rows):
        c = challenge[row]
        rr = real_response[row]
        gr = gen_response[row]

        show_image_in_ax(axs[row, 0], c, title="Challenge", use_default_v=True)
        show_image_in_ax(axs[row, 1], rr, title="Real Response")
        show_image_in_ax(axs[row, 2], gr, title="Generated Response")

    return fig


def show_image_in_ax(ax, img, title="", use_default_v=False):
    ax.axis("off")
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')

    if torch.is_tensor(img):
        img = torch.squeeze(img).cpu().numpy()

    if title:
        ax.set_title(title)

    if use_default_v:
        vmin = np.min(img)
        vmax = np.max(img)
    else:
        vmin = 0
        vmax = 1

    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)


# Expects img data to be in range [0,1]
def show_difference_map_in_ax(ax, real_response, gen_response):
    ax.axis("off")
    if torch.is_tensor(real_response):
        real_response = torch.squeeze(real_response).cpu().numpy()
        gen_response = torch.squeeze(gen_response).cpu().numpy()
        vmin = 0
        vmax = 1
    else:
        vmin = min(np.min(real_response), np.min(gen_response))
        vmax = max(np.min(real_response), np.max(gen_response))

    difference_map = np.absolute((real_response - gen_response))
    difference_map_score = np.mean(difference_map) * 255
    ax.set_title(f"Abs Diff: {difference_map_score:.2f}")
    ax.imshow(difference_map, cmap="gray", vmin=vmin, vmax=vmax)


def plot_grad_flow(named_parameters, axis):
    '''
    Plots the gradients flowing through different layers in the net during training. Assumes that a figure was
    initiated beforehand.
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    axis.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    axis.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    axis.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    axis.set_xticks(range(0, len(ave_grads), 1))
    axis.set_xticklabels(layers, rotation="vertical")
    axis.set_xlim(left=0, right=len(ave_grads))
    axis.set_ylim(bottom=-0.001, top=0.2)
    axis.set_xlabel("Layers")
    axis.set_ylabel("Average gradient")
    axis.grid(True)
    axis.legend([Line2D([0], [0], color="c", lw=4),
                 Line2D([0], [0], color="b", lw=4),
                 Line2D([0], [0], color="k", lw=4)],
                ['max-gradient', 'mean-gradient', 'zero-gradient'])


def create_grad_fig(name):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title(f"{name} Gradient flow")
    return fig, ax


def get_pred_figure(output, probability, target, number_images=10):
    probability = torch.squeeze(probability)
    predictions = torch.round(probability).int()
    # Make sure that target is an integer value so it can be used as an index later on
    target = target.int()
    cols = 5
    rows = int(math.ceil(number_images / cols))
    fig, _ = plt.subplots(rows, cols, figsize=(14, 10))

    for idx, ax in enumerate(fig.axes):
        show_image_in_ax(ax, output[idx], use_default_v=True)
        prediction = predictions[idx]
        prediction_probability = probability[idx] * 100.0
        actual_label = "Gen." if target[idx] else "Real"
        ax.set_title(
            f"Gen. (prob. {prediction_probability:.1f}%)\n(Actually: {actual_label})",
            color=("green" if prediction == target[idx].item() else "red"))
    return fig
