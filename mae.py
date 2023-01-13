from transformers import AutoFeatureExtractor, ViTMAEModel
from PIL import Image
import requests
from scipy.spatial import distance_matrix
import numpy as np
from patchify import patchify
from einops import rearrange
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

url = "http://images.cocodataset.org/val2017/000000581781.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
model.embeddings.config.mask_ratio = 0

def delta_hyp(dismat):
    """
    computes delta hyperbolicity value from distance matrix
    """

    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)

def get_rel_hyp(inputs):
    # print(inputs['pixel_values'].shape)
    images = inputs.squeeze()
    images = rearrange(images, "c w h -> w h c").numpy()
    patches = patchify(images, (16, 16, 3), step=16)
    patches = patches.reshape((14 * 14, 16 * 16 * 3))
    # print(patches)
    outputs = model(pixel_values=inputs)
    last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape)

    # print(last_hidden_states.shape)
    last_hidden_states = last_hidden_states.squeeze().detach().numpy()

    dis_mat = distance_matrix(last_hidden_states, last_hidden_states)
    delta = delta_hyp(dis_mat)
    diam = np.max(dis_mat)

    dis_p = distance_matrix(patches, patches)
    delta_p = delta_hyp(dis_p)
    diam_p = np.max(dis_p)

    return (2 * delta) / diam, (2 * delta_p) / diam_p

# print("hidden", (2 * delta) / diam)
# print("image", (2 * delta_p) / diam_p)
path = '/data/dataset/imagenet_cls_loc/CLS_LOC/ILSVRC2015/Data/CLS-LOC'

dataset_val = datasets.ImageFolder(
    path + '/val',
    lambda x: feature_extractor(images=x, return_tensors="pt")
)

val_loader = DataLoader(
    dataset_val,
    batch_size=128,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
)

total_h = 0
total_i = 0
total = 0

for images, label in val_loader:
    images = images["pixel_values"]
    for i in range(images.shape[0]):
        image = images[i]
        total += 1
        h, i = get_rel_hyp(image)
        total_h += h
        total_i += i

print("hidden", total_h/total)
print("image", total_i/total)
