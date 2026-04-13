import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_cue_batch(imgs, only_pixels=False):
    # imgs: (B, C, H, W)
    batch_size, c, h, w = imgs.shape
    cues = imgs.clone()
    masks = torch.ones((batch_size, 1, h, w), device=device)

    for i in range(batch_size):
        mask_ratio = random.uniform(0.25, 0.60)
        num_pixels = int(h * w * mask_ratio)

        if only_pixels:
            indices = torch.randperm(h * w)[:num_pixels]
            cues[i].view(c, -1)[:, indices] = 0
            masks[i].view(1, -1)[:, indices] = 0
        else:
            if random.random() > 0.5: # erase random pixels
                indices = torch.randperm(h * w)[:num_pixels]
                cues[i].view(c, -1)[:, indices] = 0
                masks[i].view(1, -1)[:, indices] = 0
            else: # erase blocks
                block_h = int(h * torch.sqrt(torch.tensor(mask_ratio)))
                block_w = int(w * torch.sqrt(torch.tensor(mask_ratio)))

                top = random.randint(0, h - block_h)
                left = random.randint(0, w - block_w)
                cues[i][:, top:top+block_h, left:left+block_w] = 0
                masks[i, :, top:top+block_h, left:left+block_w] = 0

    return cues, masks