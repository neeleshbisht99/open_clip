#%%#
"""CLIP Finetuning"""

import open_clip
print(open_clip.list_pretrained())
print('\n')

# %%
""" Download model and test"""
import torch
from PIL import Image
import open_clip


#TODO: also try : ('ViT-L-14', 'openai'), ViT-L/14-336 (openai) and ('EVA02-L-14', 'merged2b_s4b_b131k'), ('EVA02-L-14-336', 'merged2b_s6b_b61k'),
model_name = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

device_id = 6
if torch.cuda.is_available() and device_id < torch.cuda.device_count():
    device = torch.device(f"cuda:{device_id}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device).eval()

img_path = "/shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/dataset/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0024_796089.jpg"
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
text_prompts = ["a black bird flying in sky", "a black bird", "an image of sky"]
text = tokenizer(text_prompts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)


    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_idx = torch.argmax(text_probs, dim=-1).item()

print("Label probs:", text_probs)
print(f"predicted text: {text_prompts[top_idx]}")

