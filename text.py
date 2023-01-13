from transformers import AutoFeatureExtractor, ViTMAEModel, DeiTModel, AutoTokenizer, BertModel
from PIL import Image
import requests
from scipy.spatial import distance_matrix
import numpy as np
from patchify import patchify
from einops import rearrange
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch

# url = "http://images.cocodataset.org/val2017/000000581781.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
# model = ViTMAEModel.from_pretrained("facebook/vit-mae-base").cuda()
# model.embeddings.config.mask_ratio = 0

# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
# model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224").cuda()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


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
    inputs = tokenizer(inputs, return_tensors="pt")
    outputs = model(inputs)
    last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape)

    # print(last_hidden_states.shape)
    last_hidden_states = last_hidden_states.squeeze().cpu().detach().numpy()

    dis_mat = distance_matrix(last_hidden_states, last_hidden_states)
    delta = delta_hyp(dis_mat)
    diam = np.max(dis_mat)

    return (2 * delta) / diam

# print("hidden", (2 * delta) / diam)
# print("image", (2 * delta_p) / diam_p)

# dataset_val = datasets.ImageFolder(
#     path + '/val',
#     lambda x: feature_extractor(images=x, return_tensors="pt")
# )
#
# indices = torch.randperm(len(dataset_val))[:1500]
# dataset_val = Subset(dataset_val, indices)
#
# val_loader = DataLoader(
#     dataset_val,
#     batch_size=128,
#     shuffle=False,
#     num_workers=6,
#     pin_memory=True,
# )
#
# total_h = 0
# total_i = 0
# total = 0

# for images, label in tqdm(val_loader):
#     images = images["pixel_values"]
#     for i in range(images.shape[0]):
#         image = images[i]
#         total += 1
#         h, i = get_rel_hyp(image)
#         total_h += h
#         total_i += i
#
# print("hidden", total_h/total)
# print("image", total_i/total)


i = "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered controversial I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it's not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn't have much of a plot."

print(get_rel_hyp(i))
