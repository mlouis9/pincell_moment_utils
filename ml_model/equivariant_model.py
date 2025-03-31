#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# In this notebook, we train a network on the pincell moment dataset that will share certain equivariance properties incurred by the symmetry of the domain under transformations of the Diheadral group of order 8 ($D_4$). This will hopefully decrease the number of parameters required to train the model (as compared with a FCN), and encorporate some problem specific inductive bias that will aid generalization.
# 
# ## Rotational Equivariance Implies Permutation Invariance of Surfaces
# In the title
# 
# ## Reflection Equivariance Implies 180 Degree Rotation Equivariance (in the spatial and angular indices) of Input Coefficient Array
# Also in the title
# 
# ## Network Architecture
# To encode the first equivariance, we use a Graph Neural Network Architecture (GNN), and to encode the second equivariance, we use a 180 degree rotation equivariant convolutional neural network with weight tying (i.e. the filters are constrained to be invariant under 180 degree rotations).
# 
# ### CNN Architecture
# CNN architecture inspired by spatial characteristics of the Bernstein polynomials and overlap of basis functions -> local correlation in coefficient array.
# 
# Now, since the domain is not necessarily translation equivariant (though some small amount of translation equivariance may be observed in practice), we need some way to break this equivariance, and we do so by explicitly introducing some position encoding as an additional chanel in the input (3D) tensor. Importantly, to preserve the 180 degree rotation equivariance, we must ensure that this position encoding is invariant under 180 degree rotations. There are several different types of such positional encodings, so we decided to use 7 of them to ensure expressivity of translation equivariance violation.
# 
# #### Encoder Architecture
# We do this position encoding then feed the data into an upsampling equivariant CNN (transpose convolution) to create a higher resolution "image" also in a higher dimension in channel space to provide an expressive embedding. This was thought to be necessary because, to obtain adequate statistics with a computationally manageable number of neutron histories in pincell calculations, we needed to keep the expansion order small, so the coefficient arrays are only of dimension $7\times 5 \times 8$, which may not be expressive enough to capture the complexity of the neutron transport process. We also encode the surface weights as an additional, uniform, channel here.
# 
# #### Decoder Architectures
# We have technically have three encoders, all of which take the 4 surface embeddings produced at the final layer of the graph neural network as input. First, we apply a common equivariant CNN to each embedding to generate the outgoing fluxes, this also collapses the image size back to the original by downsampling with a stride length of 2. Then, at the final layer of this network, we have one layer that generates the final image (via a convolution layer), and one that generates the single scalar weight for the image (surface) by an adaptive average pooling and a sigmoid activation to ensure the weight is ∈[0,1].
# 
# Now, since, unfortunately, the pin power coefficients do not obey the same, simple, equivariance as do the surface flux expansion coefficients, we simply concatenate (channel wise) the 4 surface embeddings and pass it through a standard CNN to produce the output pin power coefficients.
# 
# Finally, since we require keff to be _invariant_ (not equivariant) under 180 degree rotations and under rotations/reflections (permutation of surfaces), we require keff to take a sum of the surface embeddings as input. We then simply map this value to a scalar by a FCNN, and apply a softplus to ensure strict positivity of keff predictions.
# 
# ### GNN Architecture
# The GNN uses the equivariant CNN in both message passing and updating. Not much else to say about this

# ## Read in the Dataset

# In[94]:


import zarr

dataset_path = '../dataset_generation/full_dataset/dataset.zarr'

# Open the dataset in read mode
root = zarr.open_group(dataset_path, mode='r')

# Access the arrays
X_flux = root["X_flux_coeffs"]
X_wts  = root["X_weights"]
Y_flux = root["Y_flux_coeffs"]
Y_pow  = root["Y_power_coeffs"]
Y_keff  = root["Y_keff"]
Y_wts = root["Y_weights"]


# ## Define the Equivariant CNN Architecture

# In[95]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def rotate180_spatial(kernel_3d):
    # kernel_3d shape: (C_out, C_in, kD, kH, kW)
    return torch.rot90(kernel_3d, k=2, dims=[-1, -2])  # rotate over (H, W)

# Reusable 3D symmetric convolution layer
class SymmetricConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=1, bias=True, transpose=False):
        super().__init__()
        self.transpose = transpose
        if transpose:
            # Transposed expects (in_channels, out_channels, ...)
            self.weight_raw = nn.Parameter(torch.randn(in_channels, out_channels, *kernel_size))
        else:
            self.weight_raw = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Permute (B, C, H, W, D) → (B, C, D, H, W)
        x = x.permute(0, 1, 4, 2, 3)
        w = 0.5 * (self.weight_raw + rotate180_spatial(self.weight_raw))

        if self.transpose:
            out = F.conv_transpose3d(x, w, bias=self.bias, stride=self.stride, padding=self.padding)
        else:
            out = F.conv3d(x, w, bias=self.bias, stride=self.stride, padding=self.padding)

        return out.permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

class SymmetricConvNet3D(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        assert num_layers >= 1, "Must have at least one layer"

        layers = []

        if num_layers == 1:
            layers.append(SymmetricConv3D(in_channels, out_channels, stride=1, transpose=False, padding=1))
        else:
            layers.append(SymmetricConv3D(in_channels, hidden_channels, stride=1, transpose=False, padding=1))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(SymmetricConv3D(hidden_channels, hidden_channels, stride=1, transpose=False, padding=1))
                layers.append(nn.ReLU())
            layers.append(SymmetricConv3D(hidden_channels, out_channels, stride=1, transpose=False, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 6:  # [B, N, C, H, W, D]
            B, N, C, H, W, D = x.shape
            x = x.reshape(B * N, C, H, W, D)  # → [B*N, C, H, W, D]
            out = self.net(x)
            out = out.reshape(B, N, *out.shape[1:])  # → [B, N, C, H, W, D]
            return out
        elif x.dim() == 5:  # [B, C, H, W, D] — single node input
            return self.net(x)
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")


# ## Define the Encoder

# In[174]:


def make_3d_position_encodings(H, W, D, batch_size):
    # Coordinate grids
    i = np.arange(H).reshape(H, 1, 1).astype(np.float32)
    j = np.arange(W).reshape(1, W, 1).astype(np.float32)
    k = np.arange(D).reshape(1, 1, D).astype(np.float32)

    # Centering
    i0 = (H - 1) / 2
    j0 = (W - 1) / 2
    k0 = (D - 1) / 2

    I = i - i0
    J = j - j0
    K = k - k0

    I_full = I + np.zeros((H, W, D), dtype=np.float32)
    J_full = J + np.zeros((H, W, D), dtype=np.float32)
    K_full = K + np.zeros((H, W, D), dtype=np.float32)

    # Normalizations
    H2, W2, D2 = H**2, W**2, D**2
    HW, HD, WD, HWD = H * W, H * D, W * D, H * W * D

    # Invariant under 180° spatial (I,J) rotation:
    r2_spatial = (I_full**2 + J_full**2) / (H2 + W2)                # radial in-plane
    r2_full = (I_full**2 + J_full**2 + K_full**2) / (H2 + W2 + D2)  # full 3D radius squared
    IJ = (I_full * J_full) / HW                                     # spatial interaction
    IK2 = (I_full**2 * K_full) / (H2 * D)                           # spatial^2 × depth
    JK2 = (J_full**2 * K_full) / (W2 * D)                           # spatial^2 × depth
    IJK = (I_full * J_full * K_full) / HWD                          # full 3-way interaction

    pos_channels = [r2_spatial, r2_full, IJ, IK2, JK2, IJK]

    pos_stack = np.stack(pos_channels, axis=0).astype(np.float32)  # (C, H, W, D)
    pos_stack_batched = np.broadcast_to(
        pos_stack[None, :, :, :, :], (batch_size, len(pos_channels), H, W, D)
    )

    return pos_stack_batched


class SpatialRotEquivariantUpsample3D(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        assert num_layers >= 1, "Must have at least one layer"

        layers = []

        if num_layers == 1:
            # Direct upsample from in → out
            layers.append(SymmetricConv3D(in_channels, out_channels, stride=2, transpose=True, padding=1))
        else:
            # First layer: upsample
            layers.append(SymmetricConv3D(in_channels, hidden_channels, stride=2, transpose=True, padding=1))
            layers.append(nn.ReLU())
            # Middle layers (if any): normal conv
            for _ in range(num_layers - 2):
                layers.append(SymmetricConv3D(hidden_channels, hidden_channels, stride=1, transpose=True, padding=1))
                layers.append(nn.ReLU())
            # Final layer: output channels, no activation
            layers.append(SymmetricConv3D(hidden_channels, out_channels, stride=1, transpose=True, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

    
class Encoder(nn.Module):
    def __init__(self, X_flux, hidden_channels, out_channels, num_layers):
        super().__init__()
        # Move axis to put channels first
        input = np.moveaxis(X_flux[:, :, :, :, :, :], -1, 2)

        self.B, self.N, self.C, self.H, self.W, self.D = input.shape
        
        # Generate symmetric positional channel
        pos_array = make_3d_position_encodings(self.H, self.W, self.D, self.B)  # NumPy array
        pos_tensor = torch.from_numpy(pos_array).float()                       # Convert to FloatTensor

        # If it's truly a constant (not trainable), register it as a buffer:
        self.register_buffer("pos_encodings", pos_tensor)


        # Note = C + 1 + 6 (positional encodings)
        in_channels = self.C + 1 + 6
        self.conv = SpatialRotEquivariantUpsample3D(in_channels, hidden_channels, out_channels, num_layers=num_layers)

    def forward(self, X_flux, X_wts):
        """
        X_flux: [B, N, H, W, D, C] (Torch tensor)
        X_wts : [B, N] or [B, N, ...] per-surface scalars
        """
        outputs = []

        for surface in range(self.N):
            # X_flux[:, surface] => shape [B, H, W, D, C]
            # We want [B, C, H, W, D], so do:
            x = X_flux[:, surface].permute(0, 4, 1, 2, 3)  # => [B, C, H, W, D]

            # Weights: shape [B], expand to [B,1,H,W,D]
            surface_wts = X_wts[:, surface].view(-1, 1, 1, 1, 1)
            surface_wts = surface_wts.expand(-1, 1, self.H, self.W, self.D)

            # Also gather pos_encodings for this batch => shape [B, 6, H, W, D]
            # Already on correct device (thanks to register_buffer) 
            # just combine along channel dimension:
            cat_input = torch.cat([x, surface_wts, self.pos_encodings], dim=1)

            out = self.conv(cat_input)
            outputs.append(out)

        # Stack across surface dimension => [B, N, out_channels, H', W', D']
        return torch.stack(outputs, dim=1)


# ## Define the Decoders

# In[97]:


class DecodeCoefficients(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        assert num_layers >= 1, "Must have at least one layer"

        layers = []
        current_in = in_channels

        for i in range(num_layers):
            is_first = (i == 0)
            is_last = (i == num_layers - 1)

            out_channels = out_channels if is_last else hidden_channels
            stride = 2 if is_first else 1

            layers.append(SymmetricConv3D(in_channels=current_in, out_channels=out_channels, stride=stride, transpose=False, padding=1))

            current_in = out_channels

            if not is_last:
                layers.append(nn.ReLU())

        self.feature_extractor = nn.Sequential(*layers)
        self.image_head = SymmetricConv3D(in_channels=out_channels, out_channels=1, stride=1, transpose=False, padding=1)
        self.scalar_head = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Linear(out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        # x: [B, N, C, I, J, D]
        B, N, C, I, J, D = x.shape
        x = x.view(B * N, C, I, J, D)

        embedding = self.feature_extractor(x)  # shape: [B*N, hidden, I', J', D']
        image_out = self.image_head(embedding)  # shape: [B*N, 1, I', J', D']

        # Permute for scalar head
        pooled = embedding.permute(0, 1, 4, 2, 3)
        scalar_out = self.scalar_head(pooled).squeeze(-1)  # shape: [B*N]

        # Now get the actual spatial dims
        _, _, I_out, J_out, D_out = image_out.shape
        image_out = image_out.view(B, N, 1, I_out, J_out, D_out)
        scalar_out = scalar_out.view(B, N)

        return image_out, scalar_out



class DecodePinPower(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_shape=(15,8)):
        super().__init__()
        self.out_shape = out_shape

        # Combine 4 tensors along channel axis: (B, 4*C, H, W, D)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=4 * in_channels, out_channels=hidden_channels, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, out_shape[0], out_shape[1]))  # Output shape: (B, C, 1, 15, 8)
        )

        input_dim = hidden_channels * out_shape[0] * out_shape[1]
        output_dim = out_shape[0] * out_shape[1]
        self.final_proj = nn.Sequential(
            nn.Flatten(start_dim=1),           # (B, hidden_channels*1*15*8)
            nn.Linear(input_dim, output_dim),  # map to 15*8
        )

    def forward(self, x_tensor):
        # x_tensor: [B, N, C, H, W, D]
        B, N, C, H, W, D = x_tensor.shape
        assert N == 4, f"Expected 4 input tensors along N, but got {N}"

        # Move N to channel axis and flatten N and C: (B, N*C, H, W, D)
        x = x_tensor.permute(0, 2, 1, 3, 4, 5)     # [B, C, N, H, W, D]
        x = x.reshape(B, N * C, H, W, D)           # [B, 4*C, H, W, D]

        out = self.net(x)                          # [B, hidden_channels, 1, 15, 8]
        out = self.final_proj(out).squeeze(1)      # [B, 15*8]
        return out.view(-1, *self.out_shape)       # [B, 15, 8]
    

class DecodeKeff(nn.Module):
    def __init__(self, in_channels=12, hidden_dim=8):
        super().__init__()
        # 1) Shrink 3D feature map (B, 12, H, W, D) to (B, 12, 1, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)  

        # 2) Map from 12 → hidden_dim → 1
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),         # (B, 12, 1, 1, 1) → (B, 12)
            nn.Linear(in_channels, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),         # single scalar 
            nn.Softplus()
        )

    def forward(self, x_tensor):
        """
        x_tensor: Tensor of shape (B, N, C, H, W, D)
                where N is the number of contributing surfaces.
        """
        # Sum over N to combine contributions: (B, C, H, W, D)
        x = x_tensor.sum(dim=1)

        # Global 3D average pooling: (B, C, 1, 1, 1)
        x = self.global_pool(x)

        # Flatten and pass through MLP: (B, 1)
        out = self.fc(x)
        return out


# ## Define GNN Architecture

# In[98]:


class EquivariantGNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_cnn_layers):
        super().__init__()
        self.message_fn = SymmetricConvNet3D(in_channels, hidden_channels, out_channels, num_cnn_layers)
        self.update_fn = SymmetricConvNet3D(in_channels + out_channels, hidden_channels, out_channels, num_cnn_layers)

    def forward(self, x):
        # x: [B, N, C, H, W, D]
        B, N, C, H, W, D = x.shape
        messages = []

        for i in range(N):
            m = 0
            for j in range(N):
                # Apply CNN to x[:, j] (shape [B, C, H, W, D])
                message = self.message_fn(x[:, j])
                m = m + message  # Sum over neighbors
            messages.append(m)

        messages = torch.stack(messages, dim=1)  # [B, N, C', H, W, D]
        updated_input = torch.cat([x, messages], dim=2)  # Concat along channels
        return self.update_fn(updated_input)


class EquivariantGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_cnn_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EquivariantGNNLayer(
                in_channels=out_channels if i > 0 else in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_cnn_layers=num_cnn_layers
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ## Now Test a Forward Pass

# In[112]:


import numpy as np

max_samples = 20

encoder = Encoder(X_flux[:max_samples], hidden_channels=8, out_channels=12, num_layers=3)

out1 = encoder.forward(X_flux[:max_samples], X_wts[:max_samples])
out2 = encoder.forward(np.rot90(X_flux[:max_samples], 2, axes=(2, 3)), X_wts[:max_samples])

gnn = EquivariantGNN(in_channels=12, hidden_channels=12, out_channels=12, num_layers=3, num_cnn_layers=3)
out1 = gnn.forward(out1)
out2 = gnn.forward(out2)

# Now decode
coef_decoder = DecodeCoefficients(in_channels=12, hidden_channels=8, out_channels=1, num_layers=3)
coefs_1, wts1 = coef_decoder.forward(out1)
coefs_2, wts2 = coef_decoder.forward(out2)

# Now decode pin power
pin_decoder = DecodePinPower(in_channels=12, hidden_channels=8, out_shape=(15,8))
pin_1 = pin_decoder.forward(out1)
pin_2 = pin_decoder.forward(out2)

# Now decode keff
keff_decoder = DecodeKeff(in_channels=12, hidden_dim=8)
keff_1 = keff_decoder.forward(out1)
keff_2 = keff_decoder.forward(out2)


# In[113]:


print(wts1[0], wts2[0])
print(keff_1[0], keff_2[0])


# In[101]:


import matplotlib.pyplot as plt

energy_idx = 2

# Rotated output
plt.imshow(np.rot90(out1[0, 0, 0].detach().numpy()[:, :, energy_idx], 2))
plt.show()
# Output of rotated input
plt.imshow(out2[0, 0, 0].detach().numpy()[:, :, energy_idx])
plt.show()


# ## Define a Full Neural Network Model

# In[165]:


import torch.nn as nn
import torch.nn.functional as F

class EquivariantModel(nn.Module):
    def __init__(self, X_flux, 
                 embedding_channels=12, 
                 num_gnn_layers=3, 
                 num_cnn_layers=2, 
                 hidden_gnn_channels=12, 
                 hidden_decoder_channels=8,
                 hidden_encoder_channels=8,
                 dropout_prob=0.1):
        super().__init__()
        
        # Your existing submodules
        self.encoder = Encoder(
            X_flux[:1], 
            hidden_channels=hidden_encoder_channels, 
            out_channels=embedding_channels, 
            num_layers=num_cnn_layers
        )
        self.gnn = EquivariantGNN(
            in_channels=embedding_channels, 
            hidden_channels=hidden_gnn_channels, 
            out_channels=embedding_channels, 
            num_layers=num_gnn_layers, 
            num_cnn_layers=num_cnn_layers
        )
        
        # Dropout3d will drop entire feature-maps across the channel dimension
        # self.mc_dropout = nn.Dropout(p=dropout_prob)
        
        self.coef_decoder = DecodeCoefficients(
            in_channels=embedding_channels, 
            hidden_channels=hidden_decoder_channels, 
            out_channels=1, 
            num_layers=num_cnn_layers
        )
        self.pin_decoder = DecodePinPower(
            in_channels=embedding_channels, 
            hidden_channels=hidden_decoder_channels, 
            out_shape=(15, 8)
        )
        self.keff_decoder = DecodeKeff(
            in_channels=embedding_channels, 
            hidden_dim=hidden_decoder_channels
        )

    def forward(self, X_flux, X_wts):
        x = self.encoder(X_flux, X_wts)  # [B, N, C, H, W, D]
        x = self.gnn(x)                  # [B, N, C, H, W, D]
        
        # Apply dropout in training mode
        x = self.mc_dropout(x)
        
        # Decode different targets
        coefs, wts   = self.coef_decoder(x)
        pin_flux     = self.pin_decoder(x)
        keff         = self.keff_decoder(x)
        return coefs, wts, pin_flux, keff


# In[169]:


max_samples = 1
model = EquivariantModel(X_flux[:max_samples])
coefs, wts, pin_flux, keff = model.forward(X_flux[:max_samples], X_wts[:max_samples])
coefs2, wts, pin_flux, keff = model.forward(np.rot90(X_flux[:max_samples], 2, axes=(2,3)), X_wts[:max_samples])


# In[170]:


plt.imshow(coefs.detach().numpy()[0, 0, 0, :, :, 0])


# In[171]:


plt.imshow(np.rot90(coefs2.detach().numpy()[0, 0, 0, :, :, 0], 2))


# In[127]:


print(keff, wts)


# ## Training

# In[143]:


from torch.utils.data import Dataset, DataLoader

class PincellMomentDataset(Dataset):
    """
    A simple Dataset that returns:
      X_flux[i], X_wts[i] -> Model inputs
      Y_flux[i], Y_pow[i], Y_keff[i], Y_wts[i] -> Ground-truth labels
    """
    def __init__(self, X_flux, X_wts, Y_flux, Y_pow, Y_keff, Y_wts):
        super().__init__()
        self.X_flux = X_flux
        self.X_wts  = X_wts
        self.Y_flux = Y_flux
        self.Y_pow  = Y_pow
        self.Y_keff = Y_keff
        self.Y_wts  = Y_wts

    def __len__(self):
        return len(self.X_flux)

    def __getitem__(self, idx):
        # Each of these is NumPy -> convert to torch.Tensor
        X_flux_item = torch.tensor(self.X_flux[idx], dtype=torch.float32)
        X_wts_item  = torch.tensor(self.X_wts[idx],  dtype=torch.float32)
        
        # For the labels, likewise
        Y_flux_item = torch.tensor(self.Y_flux[idx], dtype=torch.float32)
        Y_pow_item  = torch.tensor(self.Y_pow[idx],  dtype=torch.float32)
        Y_keff_item = torch.tensor(self.Y_keff[idx], dtype=torch.float32)
        Y_wts_item  = torch.tensor(self.Y_wts[idx],  dtype=torch.float32)
        
        return (X_flux_item, X_wts_item), (Y_flux_item, Y_pow_item, Y_keff_item, Y_wts_item)


# Train test split

# In[144]:


# Suppose you want an 80/20 split
num_samples = X_flux.shape[0]
indices = np.random.permutation(num_samples)
train_size = int(0.8 * num_samples)
train_indices = indices[:train_size]
test_indices  = indices[train_size:]

# Slice arrays accordingly
X_flux_train = X_flux[train_indices]
X_wts_train  = X_wts[train_indices]
Y_flux_train = Y_flux[train_indices]
Y_pow_train  = Y_pow[train_indices]
Y_keff_train = Y_keff[train_indices]
Y_wts_train  = Y_wts[train_indices]

X_flux_test = X_flux[test_indices]
X_wts_test  = X_wts[test_indices]
Y_flux_test = Y_flux[test_indices]
Y_pow_test  = Y_pow[test_indices]
Y_keff_test = Y_keff[test_indices]
Y_wts_test  = Y_wts[test_indices]

# Create PyTorch datasets
train_dataset = PincellMomentDataset(X_flux_train, X_wts_train, Y_flux_train, Y_pow_train, Y_keff_train, Y_wts_train)
test_dataset  = PincellMomentDataset(X_flux_test,  X_wts_test,  Y_flux_test,  Y_pow_test,  Y_keff_test,  Y_wts_test)

# Dataloaders
batch_size = 4  # adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


# Training loop

# In[175]:


# Initialize model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EquivariantModel(
    X_flux, 
    embedding_channels=12,
    num_gnn_layers=3,
    num_cnn_layers=2,
    hidden_gnn_channels=12,
    hidden_decoder_channels=8,
    hidden_encoder_channels=8,
    dropout_prob=0.1
).to(device)

# Define optimizer and MSE loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse_loss  = nn.MSELoss()

num_epochs = 10
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    ########################
    #      Training       #
    ########################
    model.train()  # Enable dropout and grad
    running_train_loss = 0.0
    
    for (X_flux_batch, X_wts_batch), (Y_flux_batch, Y_pow_batch, Y_keff_batch, Y_wts_batch) in train_loader:
        # Move data to GPU
        X_flux_batch = X_flux_batch.to(device)
        X_wts_batch  = X_wts_batch.to(device)
        Y_flux_batch = Y_flux_batch.to(device)
        Y_pow_batch  = Y_pow_batch.to(device)
        Y_keff_batch = Y_keff_batch.to(device)
        Y_wts_batch  = Y_wts_batch.to(device)

        # Forward pass
        pred_coefs, pred_wts, pred_pin_flux, pred_keff = model(X_flux_batch, X_wts_batch)

        # Compute MSE losses; shape details must align your data properly
        loss_coefs  = mse_loss(pred_coefs,  Y_flux_batch)
        loss_wts    = mse_loss(pred_wts,    Y_wts_batch)
        loss_pin    = mse_loss(pred_pin_flux, Y_pow_batch)
        loss_keff   = mse_loss(pred_keff,   Y_keff_batch)

        # Combine them (you can weight them differently)
        total_loss = loss_coefs + loss_wts + loss_pin + loss_keff

        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Accumulate for epoch average
        running_train_loss += total_loss.item() * X_flux_batch.size(0)

    epoch_train_loss = running_train_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    ########################
    #       Testing       #
    ########################
    model.eval()  # Turns off dropout by default
    running_test_loss = 0.0
    
    with torch.no_grad():
        for (X_flux_batch, X_wts_batch), (Y_flux_batch, Y_pow_batch, Y_keff_batch, Y_wts_batch) in test_loader:
            # Move data to GPU
            X_flux_batch = X_flux_batch.to(device)
            X_wts_batch  = X_wts_batch.to(device)
            Y_flux_batch = Y_flux_batch.to(device)
            Y_pow_batch  = Y_pow_batch.to(device)
            Y_keff_batch = Y_keff_batch.to(device)
            Y_wts_batch  = Y_wts_batch.to(device)

            pred_coefs, pred_wts, pred_pin_flux, pred_keff = model(X_flux_batch, X_wts_batch)

            loss_coefs  = mse_loss(pred_coefs,  Y_flux_batch)
            loss_wts    = mse_loss(pred_wts,    Y_wts_batch)
            loss_pin    = mse_loss(pred_pin_flux, Y_pow_batch)
            loss_keff   = mse_loss(pred_keff,   Y_keff_batch)

            total_loss = loss_coefs + loss_wts + loss_pin + loss_keff
            running_test_loss += total_loss.item() * X_flux_batch.size(0)

    epoch_test_loss = running_test_loss / len(test_dataset)
    test_losses.append(epoch_test_loss)

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")

