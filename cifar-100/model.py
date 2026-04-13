import torch

class Encoder(torch.nn.Module):
    def __init__(self, latent=128):
        super(Encoder, self).__init__()

        self.model = torch.nn.Sequential(
            # input: (3, 32, 32)
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # Output: (32, 16, 16)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: (64, 8, 8)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: (128, 4, 4)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),

            torch.nn.Flatten(), # Output: (128 * 4 * 4) = 2048
            torch.nn.Linear(2048, latent)
        )
    
    def forward(self, x):
        return self.model(x)


class Decoder(torch.nn.Module):
    def __init__(self, latent=128):
        super(Decoder, self).__init__()
        
        self.linear = torch.nn.Linear(latent, 2048)
        
        self.model = torch.nn.Sequential(
            # start: (128, 4, 4)
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (64, 8, 8)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (32, 16, 16)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            
            torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # (3, 32, 32)
            torch.nn.Sigmoid() # pixel values are normalized to [0, 1]
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 128, 4, 4)
        return self.model(x)


class CueEncoder(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super(CueEncoder, self).__init__()
        self.model = torch.nn.Sequential(
            # input : (3, 32, 32)
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 16x16
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x8
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 4x4
            torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 4 * 4, latent_dim) 
        )

    def forward(self, x):
        return self.model(x)


class CueDecoder(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super(CueDecoder, self). __init__()
        self.linear = torch.nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.model = torch.nn.Sequential(
            # start : (256, 4, 4)
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # 32x32
            torch.nn.Sigmoid() 
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, 256, 4, 4)
        return self.model(x)


class Comparator(torch.nn.Module):
    def __init__(self, latent=128, heads=8):
        super(Comparator, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=latent, num_heads=heads)
        
        self.norm = torch.nn.LayerNorm(latent)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, latent)
        )

    def forward(self, query_latent, memories, winner_takes_all=False):
        attn_output, weights = self.attention(query_latent, memories, memories)

        if winner_takes_all:
            winner_idx = torch.argmax(weights.squeeze(1), dim=-1) # Shape: (B)
            chosen_memories = memories[winner_idx] # Shape: (B, 1, latent)
            return chosen_memories.transpose(0, 1), weights
        else:
            refined_latent = self.norm(attn_output + query_latent)
        return self.mlp(refined_latent), weights


class Memory(torch.nn.Module):
    def __init__(self, encoder, cue_encoder, decoder, comparator):
        super(Memory, self).__init__()
        self.encoder = encoder
        self.cue_encoder = cue_encoder
        self.decoder = decoder
        self.comparator = comparator

        self.latent_memories = None

    def store_memories(self, clean_images):
        self.encoder.eval()
        with torch.no_grad():
            self.latent_memories = self.encoder(clean_images)
        self.latent_memories = self.latent_memories.unsqueeze(1)

    def forward(self, cue):
        latent_cue = self.cue_encoder(cue)
        query = latent_cue.unsqueeze(0)
        latent_memory, attn_weights = self.comparator(query, self.latent_memories)
        retrieved_image = self.decoder(latent_memory.squeeze(0))
        return retrieved_image, attn_weights    
