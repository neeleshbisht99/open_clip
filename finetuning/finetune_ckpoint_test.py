import torch
import open_clip
from PIL import Image

model_path = "/shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/finetuning/logs/cub_3cls_vitb32/checkpoints/epoch_latest.pt"

device_id = 4
if torch.cuda.is_available() and device_id < torch.cuda.device_count():
    device = torch.device(f"cuda:{device_id}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'

model, _, preprocess = open_clip.create_model_and_transforms(model_name = model_name, pretrained = model_path)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device).eval()


img_path = "/shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/dataset/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0024_796089.jpg"
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
text_prompts = ["a Black_footed_Albatross flying in sky","a black bird flying in sky", "a black bird", "an image of sky", "the bird is of black color"]
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