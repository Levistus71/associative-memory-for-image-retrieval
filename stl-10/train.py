import torch
import model
import torchvision
from torch.utils.data import DataLoader
import utils
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '/content/drive/MyDrive/memory_stl10'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_SIZE = 1024

def plot_training_results(stage1, stage2, stage3_data):
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'serif'
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.lineplot(ax=axes[0], data=stage1, color='#2ecc71', linewidth=2.5)
    axes[0].set_title("Stage 1: Autoencoder Loss", fontweight='bold')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("MSE Loss")

    sns.lineplot(ax=axes[1], data=stage2, color='#3498db', linewidth=2.5)
    axes[1].set_title("Stage 2: Cue Robustness", fontweight='bold')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("MSE Loss")

    ce, mse, total = stage3_data
    sns.lineplot(ax=axes[2], data=ce, label="Retrieval (CE)", color='#e74c3c')
    sns.lineplot(ax=axes[2], data=mse, label="Latent Refinement (MSE)", color='#f1c40f')
    sns.lineplot(ax=axes[2], data=total, label="Total Loss", color='#9b59b6', linestyle='--')
    axes[2].set_title("Stage 3: Memory Comparator", fontweight='bold')
    axes[2].set_xlabel("Epochs")
    axes[2].legend(fontsize='small', frameon=True)

    plt.tight_layout()
    plt.savefig(f"{data_path}/training_summary.png", dpi=300)
    plt.show()


def train_stage1(encoder, decoder, batch_size, epochs):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    trainset = torchvision.datasets.STL10(
        root=data_path, 
        split='unlabeled',
        download=True, 
        transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    criterion = torch.nn.MSELoss()

    encoder.train()
    decoder.train()

    print("Stage 1 training starting...")

    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            latent = encoder(inputs)
            outputs = decoder(latent)

            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        print(f"epoch : {epoch}")

    print("Finished stage 1.")
    return loss_history


def train_stage2(cue_encoder, cue_decoder, batch_size, epochs):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    trainset = torchvision.datasets.STL10(
        root=data_path, 
        split='unlabeled',
        download=True, 
        transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    params = list(cue_encoder.parameters()) + list(cue_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    criterion = torch.nn.MSELoss()

    cue_encoder.train()
    cue_decoder.train()

    loss_history = []

    print("Stage 2 training starting...")

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, _ = data
            inputs = inputs.to(device)
            inputs, masks = utils.generate_cue_batch(inputs)

            optimizer.zero_grad()

            latent = cue_encoder(inputs)
            outputs = cue_decoder(latent)

            pixel_mse = torch.nn.functional.mse_loss(outputs, inputs, reduction='none')

            loss = (pixel_mse * (masks * 20.0 + (1 - masks) * 10.0)).mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        print(f"epoch : {epoch}")

    print(f"Finished stage 2.")
    return loss_history


def train_stage3(encoder, cue_encoder, comparator, batch_size, epochs):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    trainset = torchvision.datasets.STL10(
        root=data_path,
        split='unlabeled',
        download=True,
        transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    for param in encoder.parameters():
        param.requires_grad = False
    for param in cue_encoder.parameters():
        param.requires_grad = False

    params = list(comparator.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    ce_criterion = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss()

    comparator.train()

    ce_loss_history = []
    mse_loss_history = []
    loss_history = []

    print("Stage 3 training starting...")

    for epoch in range(epochs):
        ce_running_loss = 0.0
        mse_running_loss = 0.0
        running_loss = 0.0
        for images, _ in trainloader:
            images = images.to(device)
            cues, _ = utils.generate_cue_batch(images)
            cues = cues.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                latent_mem = encoder(images)
                latent_cue = cue_encoder(cues)

            q = latent_cue.unsqueeze(1)
            k = latent_mem.unsqueeze(1)

            refined_latent, attn_weights = comparator(q, k)

            targets = torch.arange(images.size(0)).to(device)
            loss_ce = ce_criterion(attn_weights.squeeze(0) / 0.07, targets)
            loss_mse = mse_criterion(refined_latent.squeeze(1), latent_mem)

            loss = (10 * loss_ce) + loss_mse
            loss.backward()
            optimizer.step()

            ce_running_loss += loss_ce.item()
            mse_running_loss += loss_mse.item()
            running_loss += loss.item()

        avg_ce_loss = ce_running_loss / len(trainloader)
        ce_loss_history.append(avg_ce_loss)

        avg_mse_loss = mse_running_loss / len(trainloader)
        mse_loss_history.append(avg_mse_loss)

        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)

        print(f"epoch : {epoch}")

    print("Finished stage 3.")
    return ce_loss_history, mse_loss_history, loss_history


def train(to_train_stage1=True, to_train_stage2=True, to_train_stage3=True):
    # stage 1 -> encoder + decoder
    # stage 2 -> cue_encoder + cue_decoder
    # state 3 -> comparator

    encoder = model.Encoder(latent=LATENT_SIZE).to(device)
    decoder = model.Decoder(latent=LATENT_SIZE).to(device)

    cue_encoder = model.CueEncoder(latent_dim=LATENT_SIZE).to(device)
    cue_decoder = model.CueDecoder(latent_dim=LATENT_SIZE).to(device)

    comparator= model.Comparator(latent=LATENT_SIZE).to(device)

    s1, s2, s3 = [], [], ([], [], [])

    if to_train_stage1:
        s1 = train_stage1(encoder, decoder, 256, 30)
        torch.save(encoder.state_dict(), f"{data_path}/encoder_weights.pth")
        torch.save(decoder.state_dict(), f"{data_path}/decoder_weights.pth")
    else:
        encoder.load_state_dict(torch.load(f"{data_path}/encoder_weights.pth", map_location=device))
        decoder.load_state_dict(torch.load(f"{data_path}/decoder_weights.pth", map_location=device))

    if to_train_stage2:
        s2 = train_stage2(cue_encoder, cue_decoder, 256, 30)
        torch.save(cue_encoder.state_dict(), f"{data_path}/cue_encoder_weights.pth")
        torch.save(cue_decoder.state_dict(), f"{data_path}/cue_decoder_weights.pth")
    else:
        cue_encoder.load_state_dict(torch.load(f"{data_path}/cue_encoder_weights.pth", map_location=device))

    if to_train_stage3:
        s3 = train_stage3(encoder, cue_encoder, comparator, 512, 50)
        torch.save(comparator.state_dict(), f"{data_path}/comparator_weights.pth")
    else:
        comparator.load_state_dict(torch.load(f"{data_path}/comparator_weights.pth", map_location=device))
    
    if any([to_train_stage1, to_train_stage2, to_train_stage3]):
        plot_training_results(s1, s2, s3)


if __name__ == "__main__":
    if device.type!="cuda":
        print("No cuda.")
        exit(1)
    train(True, True, True)