from torch import nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=8,hidden_dim=3):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        
        self.layers = nn.Sequential(
            nn.Conv3d(2, hidden_dim, kernel_size=(2,2,2)),  
            nn.GELU(),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),# (16,100,50,50)
            
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(5,5,5)),  
            nn.GELU(),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)), #48, 23, 23

            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(9,4,4)),  
            nn.GELU(),
            nn.MaxPool3d((2, 2, 2), stride=(4, 2, 2)),  #3, 10, 10, 10]
            
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3,3,3)),  
            nn.GELU(),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            
            nn.Flatten(), #192
            nn.LayerNorm(64* hidden_dim),
            nn.Linear(64* hidden_dim, latent_dim),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dims,hidden_dim=3):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dims, 64* hidden_dim),
            nn.GELU(),
            nn.LayerNorm(64* hidden_dim),
            nn.Unflatten(dim=1, unflattened_size=( hidden_dim, 4, 4, 4)),
            
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d( hidden_dim, hidden_dim, kernel_size=(3,3,3)),  
            nn.GELU(),       ##, 10, 10, 10]
            
            nn.Upsample(scale_factor=(4,2,2)),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=(9,4,4)),  
            nn.GELU(),      #48, 23, 23

            nn.Upsample(scale_factor=(2,2,2)),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=(5,5,5)),  
            nn.GELU(),   #,100,50,50)
            
            nn.Upsample(scale_factor=(2,2,2)),
            nn.ConvTranspose3d(hidden_dim, 2, kernel_size=(2,2,2)),       
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self,latent_dims,hidden_dim=3):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims,hidden_dim=hidden_dim)
        self.decoder = Decoder(latent_dims,hidden_dim=hidden_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

