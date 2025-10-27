#%%#
import open_clip
import torch
import pandas as pd
from PIL import Image
import numpy as np

model_name = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'
csv_path = '/shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/dataset/cub_finetune_100cls_test.csv'
device_id = 6
batch_size = 128

device = torch.device(f"cuda:{device_id}")
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device).eval()

df = pd.read_csv(csv_path)


img_embs = []
cap_embs = []

with torch.no_grad():
    start = 0
    while start < len(df):
        end = min(start+batch_size, len(df))
        imgs = []
        for path in df['filepath'].iloc[start:end]:
            with Image.open(path) as im:
                img = preprocess(im.convert('RGB')).unsqueeze(0)
                imgs.append(img)    
        imgs = torch.cat(imgs).to(device)
        img_feats = model.encode_image(imgs)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_embs.append(img_feats.cpu())
        del imgs, img_feats
        torch.cuda.empty_cache()
        start+=batch_size
    
    start = 0
    while start < len(df):
        end = min(start+batch_size, len(df))
        caps = tokenizer(df['caption'].iloc[start:end].tolist()).to(device)
        cap_feats = model.encode_text(caps)
        cap_feats /= cap_feats.norm(dim=-1, keepdim=True)
        cap_embs.append(cap_feats.cpu())
        del caps, cap_feats
        torch.cuda.empty_cache()
        start+=batch_size

img_embs = torch.cat(img_embs, dim=0)
cap_embs = torch.cat(cap_embs, dim=0)

sim = img_embs @ cap_embs.T
preds = sim.argmax(dim=1)
acc = (preds == torch.arange(len(df))).float().mean().item()*100
print(f"Zero-shot accuracy: {acc:.2f}%" )

np.savez(
    'img_embs_cap_embs_zeroshot_test_laion_vitb32.npz',
    img_embs=img_embs.numpy(),
    cap_embs=cap_embs.numpy(),
    file_paths = df['filepath'].to_numpy(),
    captions = df['caption'].to_numpy() 
)
print("Saved embeddings!")

# %%
# Result


# nohup python finetuning/eval/pre_tuning.py > pre_tuning_log.out 2>&1 &