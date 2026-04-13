import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent=128):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            # Input: (3, 96, 96)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # (32, 48, 48)
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (64, 24, 24)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (128, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 6, 6)
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Flatten(), # Output: (256 * 6 * 6) = 9216
            nn.Linear(9216, latent)
        )
    
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent=128):
        super(Decoder, self).__init__()
        
        self.linear = nn.Linear(latent, 9216)
        
        self.model = nn.Sequential(
            # Start: (256, 6, 6)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (128, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (64, 24, 24)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (32, 48, 48)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # (3, 96, 96)
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 256, 6, 6)
        return self.model(x)


class CueEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CueEncoder, self).__init__()
        self.model = nn.Sequential(
            # input : (3, 96, 96)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # 48x48
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 24x24
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 12x12
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 6x6
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, latent_dim) 
        )

    def forward(self, x):
        return self.model(x)


class CueDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(CueDecoder, self). __init__()
        self.linear = nn.Linear(latent_dim, 512 * 6 * 6)
        
        self.model = nn.Sequential(
            # start : (512, 6, 6)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 48x48
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # 96x96
            nn.Sigmoid() 
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, 512, 6, 6)
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

    def forward(self, query_latent, latent_memories, winner_takes_all=False):
        query_latent = torch.nn.functional.normalize(query_latent, p=2, dim=1)
        memories = torch.nn.functional.normalize(latent_memories, p=2, dim=1)

        attn_output, weights = self.attention(query_latent, memories, memories)

        if winner_takes_all:
            winner_idx = torch.argmax(weights.squeeze(1), dim=-1) # Shape: (B)
            chosen_memories = latent_memories[winner_idx] # Shape: (B, 1, latent)
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
