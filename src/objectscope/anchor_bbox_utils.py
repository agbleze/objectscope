
# COPIED FROM: https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-Detectron2/blob/main/Chapter07/Detectron2_Chapter07_Anchors.ipynb
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
import torch
from detectron2.engine import DefaultTrainer
import math
from tqdm import tqdm
import torch

plt.rcParams["figure.dpi"] = 150


def get_gt_boxes_batch(data):
    gt_boxes = [
        item["instances"].get("gt_boxes").tensor 
        for item in data
    ]
    return torch.concatenate(gt_boxes)

def get_gt_boxes(trainer: DefaultTrainer, iterations):
    trainer._data_loader_iter = iter(trainer.data_loader)
    gt_boxes = [
        get_gt_boxes_batch(next(trainer._data_loader_iter))
        for _ in tqdm(range(iterations))
    ]
    return torch.concatenate(gt_boxes)


def boxes2wh(boxes):
    x1y1 = boxes[:, :2]
    x2y2 = boxes[:, 2:]
    return x2y2 - x1y1


def wh2size(gt_wh):
    return torch.sqrt(gt_wh[:,0] * gt_wh[:,1])

def wh2ratio(wh):
    return wh[:, 1] / wh[:, 0]


def best_ratio(ac_wh, gt_wh):
    all_ratios = gt_wh[:, None] / ac_wh[None]
    inverse_ratios = 1 / all_ratios
    ratios = torch.min(all_ratios, inverse_ratios)
    worst = ratios.min(-1).values
    best = worst.max(-1).values
    return best

def fitness(ac_wh, gt_wh, EDGE_RATIO_THRESHOLD = 0.25):
    ratio = best_ratio(ac_wh, gt_wh)
    return (ratio * (ratio > EDGE_RATIO_THRESHOLD).float()).mean()

def best_recall(ac_wh, gt_wh, EDGE_RATIO_THRESHOLD=0.25):
    ratio = best_ratio(ac_wh, gt_wh)
    best = (ratio > EDGE_RATIO_THRESHOLD).float().mean()
    return best


def estimate_clusters(values, num_clusters, iter=100):
    std = values.std(0).item()
    k, _ = kmeans(values / std, num_clusters, iter=iter)
    k *= std
    return k

def visualize_clusters(values, centers):
    plt.hist(values, histtype="step")
    plt.scatter(centers, [0]*len(centers), c="red")
    plt.show()
    

def evolve(sizes, ratios, gt_wh,
           iterations=10_000,
           probability=0.9,
           muy=1, sigma=0.05,
           fit_fn=fitness,
           verbose=False
           ):
    anchors = generate_cell_anchors(tuple(sizes), tuple(ratios))
    ac_wh = boxes2wh(anchors)
    best_fit = fit_fn(ac_wh, gt_wh)
    anchor_shape = len(sizes) + len(ratios)
    
    pbar = tqdm(range(iterations), desc="Evolving ratios and sizes:")
    
    for i, _ in enumerate(pbar):
        mutation = np.ones(anchor_shape)
        mutate = np.random.random(anchor_shape) < probability
        mutation = np.random.normal(muy, sigma, anchor_shape)*mutate
        mutation = mutation.clip(0.3, 3.0)
        # mutated
        mutated_sizes = sizes.copy()*mutation[:len(sizes)]
        mutated_ratios = ratios.copy()*mutation[-len(ratios):]
        mutated_anchors = generate_cell_anchors(tuple(mutated_sizes), tuple(mutated_ratios))
        mutated_ac_wh = boxes2wh(mutated_anchors)
        mutated_fit = fit_fn(mutated_ac_wh, gt_wh)
        
        if mutated_fit > best_fit:
            sizes = mutated_sizes.copy()
            ratios = mutated_ratios.copy()
            best_fit = mutated_fit
            pbar.desc = (f"Evolving {ratios} and {sizes}, Fitness = {best_fit: .4f}")
            
    return sizes, ratios


# COPIED FROM: Detectron2 source code detectron2/modeling/anchor_generator.py
def generate_cell_anchors(sizes=(32, 64, 128, 256, 512), 
                          aspect_ratios=(0.5, 1, 2)
                          ):
    """
    Generate a tensor storing canonical anchor boxes, which are all anchor
    boxes of different sizes and aspect_ratios centered at (0, 0).
    We can later build the set of anchors for a full feature map by
    shifting and tiling these tensors (see `meth:_grid_anchors`).

    Args:
        sizes (tuple[float]):
        aspect_ratios (tuple[float]]):

    Returns:
        Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
            in XYXY format.
    """
    anchors = []
    for size in sizes:
        area = size**2.0
        for aspect_ratio in aspect_ratios:
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
    return torch.tensor(anchors)


def get_anchor_sizes_ratios(trainer: DefaultTrainer, iterations=1000,):
    gt_boxes = get_gt_boxes(trainer, 1000)
    gt_wh = boxes2wh(gt_boxes)
    e_sizes, e_ratios = evolve(gt_sizes, gt_ratios, gt_wh, 
                           iterations=1_000
                           )
    