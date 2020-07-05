# 3rd party
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import widgets
from scipy.ndimage import gaussian_filter
import numpy as np
import torch


class DrawingPad:
    """ Interactive drawing pad which implements a grid of cells, to which a
    user can draw using mouse. The drawn image is processed by `model` which
    returns the predicted class. The drawn image can be blurred by gaussian
    blurring before the prediction is run. The buttons "reset", "blur" and
    "predict" are implemented.

    Args:
        shape_grid (tuple): Shape of the grid (H, W) [px].
        model (nn.Module):  Neural network used to make predictions.
        device: Torch device.

    """

    def __init__(self, shape_grid, model, device):

        # Vis. and mouse control flags.
        self._initialized = False
        self._down = False
        # self._predicted = False

        # Model used to make predictions, device used to convert arrays to tensors
        self.model = model
        self.device = device

        # Data, cells
        self._cells = np.zeros(shape_grid, dtype=np.float32)

        # Create figure, figure manager, configure axes.
        fig = plt.figure(figsize=(4, 4))  # (figsize=(1, w / h), dpi=h)
        self._figmngr = plt.get_current_fig_manager()
        self._ax = fig.gca()
        self._ax.set_xlim(0, shape_grid[1])
        self._ax.set_ylim(shape_grid[0], 0)
        self._ax.set_xticks([])
        self._ax.set_yticks([])

        # Image canvas object.
        self._img_obj = self._ax.imshow(self._cells, cmap='gray')

        # Gaussian blur sigma, brightness multiplier
        self._SIGMA = 1
        self._BRI_MUL = 2

        # Mouse callbacks.
        plt.connect('motion_notify_event', self._on_move)
        plt.connect('button_press_event', self._on_press)
        plt.connect('button_release_event', self._on_release)

        # Reset button.
        rbutton = widgets.Button(description='reset')
        rbutton.on_click(self._on_reset_button_click)
        display(rbutton)

        # Blur button.
        pbutton = widgets.Button(description='blur')
        pbutton.on_click(self._on_blur_button_click)
        display(pbutton)

        # Predict button.
        pbutton = widgets.Button(description='predict')
        pbutton.on_click(self._on_predict_button_click)
        display(pbutton)

    @property
    def grid(self):
        """ Getter. """
        return self._cells

    def _reset(self):
        """ Sets the underlying data matrix to 0, initializes the object. """
        self._cells[:] = 0.
        self._initialized = False
        # self._predicted = False
        self._img_obj.set_data(self._cells)
        self._figmngr.canvas.draw_idle()

    def __getitem__(self, key):
        return self._cells[key[0], key[1]]

    def __setitem__(self, key, val):
        r, c = np.array(key)
        self._cells[r, c] = val

    def _draw(self):
        self._img_obj.set_data(self._cells)
        self._figmngr.canvas.draw_idle()

        # Due to pyplot bug, self._img_obj must be set twice.
        if not self._initialized:
            self._initialized = True
            self._img_obj = self._ax.imshow(self._cells, cmap='gray')

    def _on_move(self, event):
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

        # Only draw if left button pressed.
        # if self._down and not self[y, x]:
        if self._down:
            self[y, x] = 1.
            self._draw()

    def _on_press(self, event):
        # self._down = not self._predicted
        self._down = True

    def _on_release(self, event):
        self._down = False

    def _on_reset_button_click(self, b):
        self._reset()

    def _gaussian_blur_cells(self):
        self._cells = gaussian_filter(self._cells, sigma=self._SIGMA) * self._BRI_MUL

    def _on_predict_button_click(self, b):
        # self._predicted = True

        # Make prediction
        inp = torch.from_numpy(self._cells).to(self.device)[None, None]
        pred = self.model(inp)
        clp = torch.argmax(pred)
        print("\rPrediction: {}".format(clp.item()), end='')

    def _on_blur_button_click(self, b):
        self._gaussian_blur_cells()
        self._draw()

def visualize_convolution():
    fig1 = plt.figure(figsize=(8, 4))

    x = np.array([2,2,2,2,2,2,2,10,10,10,10,10,1,1,1,1,1,1,1,1,5,5,5,5,5])
    h = np.array([-1,0,1])
    y = np.convolve(a=x,v=h,mode='same')

    for i in range(-1,-1+y.shape[0]):
        fig1.clear()

        plt.subplot(1,2,1)
        markerline, stemlines, baseline = plt.stem(x, linefmt=':', markerfmt="*", label="input", use_line_collection=True)
        plt.setp(stemlines, color='r', linewidth=2)
        plt.stem(range(i,i+3), h[::-1], label="filter", use_line_collection=True)
        plt.title("input signal and filter")
        plt.legend()

        plt.subplot(1,2,2)
        plt.stem(y[0:(i+2)], label="result", use_line_collection=True)
        plt.title("result")
        plt.legend()

        plt.suptitle("convolution visualization")

        fig1.canvas.draw()
        user_input = input()

        if (user_input == "q" or user_input == "Q"):
            break

def accuracy(x, y):
    """ Accuracy.

    Args:
        x (torch.Tensor of float32): Predictions (logits), shape (B, C), B is
            batch size, C is num classes.
        y (torch.Tensor of int64): GT labels, shape (B, ),
            B = {b: b \in {0 .. C-1}}.

    Returns:
        Accuracy, in [0, 1].
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return np.mean(np.argmax(x, axis=1) == y)

def load_blackwhite_image(image_name):
    image = np.mean(plt.imread(image_name), axis=2)
    return image
