from transformers import AutoFeatureExtractor, ViTMAEModel
from PIL import Image
import requests
from scipy.spatial import distance_matrix
import numpy as np

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

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

print((2 * delta) / diam)
