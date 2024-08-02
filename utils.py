import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
import random
import colorsys
import pathlib
from lifelines.statistics import logrank_test

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
	if args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
	return optimizer
def print_network(net):
	num_params = 0
	num_params_train = 0
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)

def calculate_error(Y_hat, Y):
	error = 1. - np.sum(Y_hat==Y)/len(Y)
	return error
def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
def random_colors(N, bright=True):
	"""Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors

def visualize_instances_dict(
    input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours.

    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)} ,
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    """
    overlay = np.copy((input_image))

    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        cv2.drawContours(overlay, [np.array(inst_contour)], -1, inst_colour, line_thickness)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay

def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with the `ext` such as `ext='.png'`.

    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.

    Returns:
        file_path_list (list): sorted list of filepaths.
    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

class FocalLoss(nn.Module):
    def __init__(self, alpha=[1, 2, 2, 1, 1], gamma=2, num_classes=5, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
        """
        focal_loss
        :param preds:   size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  size:[B,N] or [B]
        :return:
        """

        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def benjamini_hochberg_adjusted(p_values):
    """
    Calculate the Benjamini-Hochberg adjusted p-values.

    Parameters:
    p_values (array-like): List or array of p-values from multiple hypothesis tests.

    Returns:
    array: Adjusted p-values.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    adjusted_p_values = np.zeros(n)
    for i in range(n):
        adjusted_p_values[i] = sorted_p_values[i] * n / (i + 1)

    # Ensure that adjusted p-values are monotonic
    adjusted_p_values = np.minimum.accumulate(adjusted_p_values[::-1])[::-1]

    # Clip p-values at 1
    adjusted_p_values = np.clip(adjusted_p_values, 0, 1)

    # Return the adjusted p-values in the original order
    return adjusted_p_values[np.argsort(sorted_indices)]


def find_optimal_cutpoint(data, time_col, event_col, variable_col):
    """
    Find the optimal cutpoint for a continuous variable in Kaplan-Meier analysis.

    Parameters:
    data (DataFrame): The dataset containing the time, event, and variable columns.
    time_col (str): The name of the time column.
    event_col (str): The name of the event column.
    variable_col (str): The name of the continuous variable column.

    Returns:
    float: The optimal cutpoint value.
    """
    unique_values = np.sort(data[variable_col].unique())
    best_cutpoint = None
    best_statistic = -np.inf

    for cutpoint in unique_values:
        group1 = data[data[variable_col] <= cutpoint]
        group2 = data[data[variable_col] > cutpoint]

        if len(group1) == 0 or len(group2) == 0:
            continue

        results = logrank_test(
            group1[time_col], group2[time_col],
            event_observed_A=group1[event_col],
            event_observed_B=group2[event_col]
        )

        if results.test_statistic > best_statistic:
            best_statistic = results.test_statistic
            best_cutpoint = cutpoint

    return best_cutpoint