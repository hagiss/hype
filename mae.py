from transformers import AutoFeatureExtractor, ViTMAEModel
from PIL import Image
import requests
from scipy.spatial import distance_matrix
import numpy as np
from patchify import patchify
from einops import rearrange

url = "http://images.cocodataset.org/val2017/000000581781.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
model.embeddings.config.mask_ratio = 0

inputs = feature_extractor(images=image, return_tensors="pt")
# print(inputs['pixel_values'].shape)
images = inputs['pixel_values'].squeeze()
images = rearrange(images, "c w h -> w h c").numpy()
patches = patchify(images, (16, 16, 3), step=16)
patches = patches.reshape((14*14, 16*16*3))
# print(patches)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)

# print(last_hidden_states.shape)
last_hidden_states = last_hidden_states.squeeze().detach().numpy()

dis_mat = distance_matrix(last_hidden_states, last_hidden_states)

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

delta = delta_hyp(dis_mat)
diam = np.max(dis_mat)

dis_p = distance_matrix(patches, patches)
delta_p = delta_hyp(dis_p)
diam_p = np.max(dis_p)

print("hidden", (2 * delta) / diam)
print("image", (2 * delta_p) / diam_p)
