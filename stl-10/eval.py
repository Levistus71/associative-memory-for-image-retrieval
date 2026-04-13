import torch
import model
import torchvision
from torch.utils.data import DataLoader
import utils
import matplotlib.pyplot as plt
import numpy as np

data_path = '/content/drive/MyDrive/memory_stl10'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_SIZE = 1024

def check_encoder_decoder(n=10):
    encoder = model.Encoder(latent=LATENT_SIZE).to(device)
    encoder.load_state_dict(torch.load(f"{data_path}/encoder_weights.pth", map_location=device))

    decoder = model.Decoder(latent=LATENT_SIZE).to(device)
    decoder.load_state_dict(torch.load(f"{data_path}/decoder_weights.pth", map_location=device))

    encoder.eval()
    decoder.eval()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=n, shuffle=True)

    images, _ = next(iter(testloader))
    images = images.to(device)

    with torch.no_grad():
        latent = encoder(images)
        reconstructed = decoder(latent)

    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")

        axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Reconstructed")

    plt.suptitle("Stage 1: Encoder-Decoder Check")
    plt.show()


def check_cueEncoder_cueDecoder(n=10):
    cue_encoder = model.CueEncoder(latent_dim=LATENT_SIZE).to(device)
    cue_encoder.load_state_dict(torch.load(f"{data_path}/cue_encoder_weights.pth", map_location=device))

    cue_decoder = model.CueDecoder(latent_dim=LATENT_SIZE).to(device)
    cue_decoder.load_state_dict(torch.load(f"{data_path}/cue_decoder_weights.pth", map_location=device))

    cue_encoder.eval()
    cue_decoder.eval()
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=n, shuffle=True)
    
    images, _ = next(iter(testloader))
    cues, _ = utils.generate_cue_batch(images)
    cues = cues.to(device)
    
    with torch.no_grad():
        latent = cue_encoder(cues)
        reconstructed = cue_decoder(latent)
    
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        axes[0, i].imshow(cues[i].cpu().permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Corrupted Cue")
        
        # Reconstructed from Cue
        axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Reconstructed")
        
    plt.suptitle("Stage 2: Cue Encoder-Decoder Check")
    plt.show()


def check_comparator(n=10, winner_takes_all=False):
    encoder = model.Encoder(latent=LATENT_SIZE).to(device)
    encoder.load_state_dict(torch.load(f"{data_path}/encoder_weights.pth", map_location=device))

    decoder = model.Decoder(latent=LATENT_SIZE).to(device)
    decoder.load_state_dict(torch.load(f"{data_path}/decoder_weights.pth", map_location=device))

    cue_encoder = model.CueEncoder(latent_dim=LATENT_SIZE).to(device)
    cue_encoder.load_state_dict(torch.load(f"{data_path}/cue_encoder_weights.pth", map_location=device))

    comparator = model.Comparator(latent=LATENT_SIZE).to(device)
    comparator.load_state_dict(torch.load(f"{data_path}/comparator_weights.pth", map_location=device))

    encoder.eval()
    decoder.eval()
    cue_encoder.eval()
    comparator.eval()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=transform)

    memory_loader = DataLoader(testset, batch_size=256, shuffle=True)
    mem_images, _ = next(iter(memory_loader))
    mem_images = mem_images.to(device)

    with torch.no_grad():
        latent_memories = encoder(mem_images) # Shape: (256, 128)
        latent_memories = latent_memories.unsqueeze(1)

    test_indices = torch.randperm(256)[:n]
    target_images = mem_images[test_indices]

    cues, _ = utils.generate_cue_batch(target_images)
    cues = cues.to(device)

    with torch.no_grad():
        latent_cues = cue_encoder(cues) # Shape: (n, 128)

        query = latent_cues.unsqueeze(1)
        refined_latent, attn_weights = comparator(query, latent_memories, winner_takes_all)

        reconstructed_images = decoder(refined_latent.squeeze(0))

    fig, axes = plt.subplots(3, n, figsize=(15, 6))
    for i in range(n):
        axes[0, i].imshow(target_images[i].cpu().permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Target Memory")

        axes[1, i].imshow(cues[i].cpu().permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Input Cue")

        axes[2, i].imshow(reconstructed_images[i].cpu().permute(1, 2, 0))
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("Retrieved Output")

    plt.suptitle("Stage 3: Comparator Check")
    plt.tight_layout()
    plt.show()


def evaluate_memory_performance(num_banks=20, num_memories=256):
    encoder = model.Encoder(latent=LATENT_SIZE).to(device).eval()
    decoder = model.Decoder(latent=LATENT_SIZE).to(device).eval()
    cue_encoder = model.CueEncoder(latent_dim=LATENT_SIZE).to(device).eval()
    comparator = model.Comparator(latent=LATENT_SIZE).to(device).eval()

    encoder.load_state_dict(torch.load(f"{data_path}/encoder_weights.pth", map_location=device))
    decoder.load_state_dict(torch.load(f"{data_path}/decoder_weights.pth", map_location=device))
    cue_encoder.load_state_dict(torch.load(f"{data_path}/cue_encoder_weights.pth", map_location=device))
    comparator.load_state_dict(torch.load(f"{data_path}/comparator_weights.pth", map_location=device))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=num_memories, shuffle=True)

    accuracies = []
    mse_losses = []
    ce_losses = []

    ce_criterion = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss()

    print(f"Starting evaluation across {num_banks} memory banks...")

    with torch.no_grad():
        for i, (images, _) in enumerate(testloader):
            if i >= num_banks: break
            
            images = images.to(device)
            cues, _ = utils.generate_cue_batch(images)
            
            latent_memories = encoder(images).unsqueeze(1) # (256, 1, latent)
            latent_cues = cue_encoder(cues).unsqueeze(1) # (256, 1, latent)
            refined_latent, attn_weights = comparator(latent_cues, latent_memories)
            
            weights = attn_weights.squeeze(0) 
            predictions = torch.argmax(weights, dim=-1)
            targets = torch.arange(images.size(0)).to(device)
            correct = (predictions == targets).sum().item()
            accuracy = (correct / images.size(0)) * 100
            
            loss_ce = ce_criterion(weights / 0.07, targets)
            
            reconstructed = decoder(refined_latent.squeeze(1))
            loss_mse = mse_criterion(reconstructed, images)

            accuracies.append(accuracy)
            ce_losses.append(loss_ce.item())
            mse_losses.append(loss_mse.item())

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Memory Bank ID')
    ax1.set_ylabel('Top-1 Accuracy (%)', color=color)
    ax1.plot(accuracies, color=color, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx() 
    color_mse = 'tab:red'
    color_ce = 'tab:green'
    ax2.set_ylabel('Loss Scale', color='black')
    ax2.plot(mse_losses, color=color_mse, linestyle='--', label='Image MSE')
    ax2.plot(ce_losses, color=color_ce, linestyle=':', label='Retrieval CE')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title(f"Model Performance across {num_banks} Memory Banks (Size {LATENT_SIZE}) and {num_memories} stored Images.")
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

    return accuracies, mse_losses, ce_losses



def plot_accuracy_scaling(memory_sizes=[256, 512, 1024, 2048, 4096, 8192]):
    encoder = model.Encoder(latent=LATENT_SIZE).to(device).eval()
    cue_encoder = model.CueEncoder(latent_dim=LATENT_SIZE).to(device).eval()
    comparator = model.Comparator(latent=LATENT_SIZE).to(device).eval()
    
    encoder.load_state_dict(torch.load(f"{data_path}/encoder_weights.pth", map_location=device))
    cue_encoder.load_state_dict(torch.load(f"{data_path}/cue_encoder_weights.pth", map_location=device))
    comparator.load_state_dict(torch.load(f"{data_path}/comparator_weights.pth", map_location=device))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    avg_accuracies = []

    print("Benchmarking memory scaling...")
    with torch.no_grad():
        for size in memory_sizes:
            testset = torchvision.datasets.STL10(root=data_path, split='test', transform=transform)
            testloader = DataLoader(testset, batch_size=size, shuffle=True)
            
            bank_accuracies = []
            for i, (images, _) in enumerate(testloader):
                
                images = images.to(device)
                cues, _ = utils.generate_cue_batch(images)
                
                latent_memories = encoder(images).unsqueeze(1) 
                latent_cues = cue_encoder(cues).unsqueeze(1)
                
                _, attn_weights = comparator(latent_cues, latent_memories)
                
                predictions = torch.argmax(attn_weights.squeeze(0), dim=-1)
                targets = torch.arange(images.size(0)).to(device)
                acc = (predictions == targets).float().mean().item() * 100
                bank_accuracies.append(acc)
            
            mean_acc = np.mean(bank_accuracies)
            avg_accuracies.append(mean_acc)
            print(f"Bank Size {size:4d} | Mean Accuracy: {mean_acc:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(memory_sizes, avg_accuracies, marker='s', linestyle='-', color='#3498db', linewidth=2)
    
    plt.title("Memory Retrieval Scaling: Top-1 Accuracy vs. Bank Size", fontweight='bold')
    plt.xlabel("Number of Stored Memories (Images in Bank)")
    plt.ylabel("Mean Retrieval Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log', base=2)
    plt.xticks(memory_sizes, labels=[str(s) for s in memory_sizes])
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f"{data_path}/accuracy_scaling_plot.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    if device.type!="cuda":
        print("No cuda.")
        exit(1)
    check_encoder_decoder()
    check_cueEncoder_cueDecoder()
    check_comparator(n=10, winner_takes_all=True)
    check_comparator(n=10, winner_takes_all=False)
    for i in [256,512,1024,2048]:
        evaluate_memory_performance(num_banks=20, num_memories=i)
    plot_accuracy_scaling()