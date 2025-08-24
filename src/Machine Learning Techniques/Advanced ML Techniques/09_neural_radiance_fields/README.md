# Neural Radiance Fields (NeRF)

## Overview
Neural Radiance Fields (NeRF) represent a revolutionary approach to 3D scene representation and novel view synthesis. Instead of traditional explicit 3D representations like meshes or point clouds, NeRF represents scenes as continuous neural functions that map 3D coordinates and viewing directions to volume density and view-dependent emitted radiance. This enables photorealistic novel view synthesis from sparse 2D images by learning the underlying 3D structure and appearance of scenes.

## Core Concepts

### Volume Rendering
NeRF uses volume rendering to synthesize images by integrating along rays through 3D space:

```
C(r) = ∫₀ᵗ T(t)σ(r(t))c(r(t), d)dt
```

Where:
- `C(r)` is the color of ray r
- `T(t) = exp(-∫₀ᵗ σ(r(s))ds)` is the accumulated transmittance
- `σ(r(t))` is the volume density at point r(t)
- `c(r(t), d)` is the view-dependent color at point r(t) in direction d

**Key Insight:** The neural network learns to predict density and color at every point in 3D space, enabling continuous scene representation.

### Positional Encoding
To capture high-frequency details, NeRF uses positional encoding to transform input coordinates:

```
γ(p) = (sin(2⁰πp), cos(2⁰πp), ..., sin(2ᴸ⁻¹πp), cos(2ᴸ⁻¹πp))
```

Where:
- `p` is the input coordinate (x, y, z, θ, φ)
- `L` is the number of frequency bands
- This encoding enables the MLP to represent high-frequency functions

### View-Dependent Effects
NeRF captures view-dependent effects by conditioning the color prediction on viewing direction:

```
c(x, d) = f(x, d)  # Color depends on both position and direction
```

This enables modeling of specular reflections, transparency, and other view-dependent phenomena.

### Hierarchical Sampling
NeRF uses a two-stage sampling strategy:
1. **Coarse sampling**: Uniform sampling along rays
2. **Fine sampling**: Importance sampling based on coarse predictions

This improves efficiency by focusing computation on regions with high density.

## Bizarre and Advanced Aspects

### 1. Continuous Scene Representation
NeRF represents entire 3D scenes as a single continuous neural function, unlike traditional discrete representations (meshes, voxels, point clouds).

### 2. Implicit Geometry
The geometry is implicitly defined through the density field σ(x), eliminating the need for explicit surface representations.

### 3. View-Dependent Rendering
NeRF can model complex view-dependent effects like specular reflections, transparency, and subsurface scattering through a single neural network.

### 4. Photorealistic Novel View Synthesis
NeRF can generate photorealistic images from viewpoints not seen during training, with accurate lighting, shadows, and reflections.

### 5. Sparse View Reconstruction
NeRF can reconstruct 3D scenes from very few input images (sometimes just 2-3 views), leveraging learned priors about scene structure.

### 6. Neural Volume Rendering
The entire rendering pipeline is differentiable, enabling end-to-end training from 2D images without 3D supervision.

## Technical Architecture

### NeRF Network Architecture
```python
class NeRF(nn.Module):
    def __init__(self, pos_enc_dim=10, view_enc_dim=4, hidden_dim=256):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.view_enc_dim = view_enc_dim
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(pos_enc_dim)
        self.view_encoding = PositionalEncoding(view_enc_dim)
        
        # MLP for density and color
        self.mlp = nn.Sequential(
            nn.Linear(3 + 2 * 3 * pos_enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 + 64)  # density + feature vector
        )
        
        # View-dependent color head
        self.color_head = nn.Sequential(
            nn.Linear(64 + 2 * 3 * view_enc_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x, d):
        # Encode positions and directions
        x_enc = self.pos_encoding(x)
        d_enc = self.view_encoding(d)
        
        # Forward through MLP
        h = self.mlp(x_enc)
        sigma = F.relu(h[..., 0])  # Density
        feature = h[..., 1:]       # Feature vector
        
        # View-dependent color
        color_input = torch.cat([feature, d_enc], dim=-1)
        color = self.color_head(color_input)
        
        return sigma, color
```

### Positional Encoding
```python
class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
    
    def forward(self, x):
        # x: (..., 3) -> (..., 3 * 2 * L)
        encodings = [x]
        
        for i in range(self.L):
            for fn in [torch.sin, torch.cos]:
                encodings.append(fn((2 ** i) * torch.pi * x))
        
        return torch.cat(encodings, dim=-1)
```

### Volume Rendering
```python
def volume_rendering(sigma, color, z_vals, rays_d):
    # sigma: (N_rays, N_samples)
    # color: (N_rays, N_samples, 3)
    # z_vals: (N_rays, N_samples)
    # rays_d: (N_rays, 3)
    
    # Compute distances between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)
    
    # Compute alpha values
    alpha = 1. - torch.exp(-sigma * dists)
    
    # Compute transmittance
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]
    T = torch.cat([torch.ones_like(T[:, :1]), T], dim=-1)
    
    # Compute weights
    weights = alpha * T
    
    # Render color
    rgb = torch.sum(weights[..., None] * color, dim=-2)
    
    # Render depth
    depth = torch.sum(weights * z_vals, dim=-1)
    
    # Render opacity
    opacity = torch.sum(weights, dim=-1)
    
    return rgb, depth, opacity, weights
```

### Hierarchical Sampling
```python
def hierarchical_sampling(rays_o, rays_d, z_vals_coarse, weights_coarse, N_fine):
    # Sample fine points based on coarse weights
    z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
    
    # Sample from piecewise constant PDF
    weights_coarse = weights_coarse + 1e-5  # Prevent zero weights
    pdf = weights_coarse / torch.sum(weights_coarse, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Sample fine points
    u = torch.rand(list(cdf.shape[:-1]) + [N_fine], device=cdf.device)
    u = u.contiguous()
    
    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    
    # Gather values
    cdf_g = torch.gather(cdf, -1, inds_g)
    z_vals_g = torch.gather(z_vals_coarse, -1, inds_g)
    
    # Linear interpolation
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_vals_fine = z_vals_g[..., 0] + t * (z_vals_g[..., 1] - z_vals_g[..., 0])
    
    return z_vals_fine
```

## Implementation Details

### Ray Generation
```python
def get_rays(H, W, focal, c2w):
    # Generate ray origins and directions
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H))
    i = i.t().to(c2w.device)
    j = j.t().to(c2w.device)
    
    # Normalized device coordinates
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    
    # Rotate ray directions from camera frame to world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    
    # Translate camera frame's origin to world frame
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d
```

### Training Loop
```python
def train_nerf(model, rays_o, rays_d, target_rgb, N_samples=64, N_importance=128):
    # Coarse sampling
    z_vals_coarse = torch.linspace(2, 6, N_samples).expand(rays_o.shape[0], N_samples)
    z_vals_coarse = z_vals_coarse + torch.rand_like(z_vals_coarse) * 0.01
    
    # Sample points along rays
    pts_coarse = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_coarse[..., :, None]
    
    # Query NeRF
    sigma_coarse, color_coarse = model(pts_coarse.reshape(-1, 3), 
                                      rays_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3))
    sigma_coarse = sigma_coarse.reshape(rays_o.shape[0], N_samples)
    color_coarse = color_coarse.reshape(rays_o.shape[0], N_samples, 3)
    
    # Volume rendering
    rgb_coarse, _, _, weights_coarse = volume_rendering(sigma_coarse, color_coarse, z_vals_coarse, rays_d)
    
    # Fine sampling
    z_vals_fine = hierarchical_sampling(rays_o, rays_d, z_vals_coarse, weights_coarse, N_importance)
    z_vals = torch.cat([z_vals_coarse, z_vals_fine], dim=-1)
    z_vals, _ = torch.sort(z_vals, dim=-1)
    
    # Sample fine points
    pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    # Query NeRF
    sigma_fine, color_fine = model(pts_fine.reshape(-1, 3),
                                  rays_d.unsqueeze(1).expand(-1, N_samples + N_importance, -1).reshape(-1, 3))
    sigma_fine = sigma_fine.reshape(rays_o.shape[0], N_samples + N_importance)
    color_fine = color_fine.reshape(rays_o.shape[0], N_samples + N_importance, 3)
    
    # Volume rendering
    rgb_fine, _, _, _ = volume_rendering(sigma_fine, color_fine, z_vals, rays_d)
    
    # Loss
    loss_coarse = F.mse_loss(rgb_coarse, target_rgb)
    loss_fine = F.mse_loss(rgb_fine, target_rgb)
    loss = loss_coarse + loss_fine
    
    return loss, rgb_fine
```

### Rendering Function
```python
def render_rays(model, rays_o, rays_d, near=2, far=6, N_samples=64, N_importance=128):
    # Coarse sampling
    t_vals = torch.linspace(0, 1, N_samples)
    z_vals = near * (1 - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[0], N_samples)
    
    # Add noise for training
    if model.training:
        z_vals = z_vals + torch.rand_like(z_vals) * (far - near) / N_samples
    
    # Sample points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    # Query NeRF
    sigma, color = model(pts.reshape(-1, 3), 
                        rays_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3))
    sigma = sigma.reshape(rays_o.shape[0], N_samples)
    color = color.reshape(rays_o.shape[0], N_samples, 3)
    
    # Volume rendering
    rgb, depth, opacity, weights = volume_rendering(sigma, color, z_vals, rays_d)
    
    return rgb, depth, opacity, weights
```

## Advanced Variants

### 1. Instant-NGP (Neural Graphics Primitives)
Uses hash-based positional encoding for faster training:

```python
class HashEncoding(nn.Module):
    def __init__(self, num_levels=16, max_hash=2**14, base_resolution=16):
        super().__init__()
        self.num_levels = num_levels
        self.max_hash = max_hash
        self.base_resolution = base_resolution
        
        # Hash tables for each level
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(2**min(i + base_resolution, 14), 2) * 1e-4)
            for i in range(num_levels)
        ])
    
    def forward(self, x):
        # x: (N, 3) normalized to [0, 1]
        encodings = []
        
        for i, hash_table in enumerate(self.hash_tables):
            # Scale coordinates
            scale = 2**i
            scaled_x = x * scale
            
            # Hash coordinates
            hashed = self.hash3d(scaled_x)
            
            # Lookup in hash table
            encoding = hash_table[hashed % len(hash_table)]
            encodings.append(encoding)
        
        return torch.cat(encodings, dim=-1)
    
    def hash3d(self, x):
        # Simple spatial hash function
        return (x[..., 0] * 73856093 + x[..., 1] * 19349663 + x[..., 2] * 83492791).long()
```

### 2. NeRF with External Memory
Incorporates external memory for better representation:

```python
class MemoryNeRF(nn.Module):
    def __init__(self, memory_size=1024, memory_dim=256):
        super().__init__()
        self.nerf = NeRF()
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_attention = nn.MultiheadAttention(memory_dim, num_heads=8)
    
    def forward(self, x, d):
        # Query NeRF
        sigma, color = self.nerf(x, d)
        
        # Read from memory
        x_features = self.nerf.mlp[:-1](x)  # Get features before final layer
        memory_read, _ = self.memory_attention(x_features, self.memory, self.memory)
        
        # Combine with NeRF output
        enhanced_features = x_features + memory_read
        enhanced_output = self.nerf.mlp[-1](enhanced_features)
        
        return enhanced_output[..., 0], enhanced_output[..., 1:]
```

### 3. Dynamic NeRF
Models scenes that change over time:

```python
class DynamicNeRF(nn.Module):
    def __init__(self, time_encoding_dim=4):
        super().__init__()
        self.nerf = NeRF()
        self.time_encoding = PositionalEncoding(time_encoding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * 3 * time_encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, x, d, t):
        # Encode time
        t_enc = self.time_encoding(t)
        t_features = self.time_mlp(t_enc)
        
        # Query NeRF with time conditioning
        sigma, color = self.nerf(x, d)
        
        # Modulate output with time features
        sigma = sigma * (1 + t_features[..., 0])
        color = color * (1 + t_features[..., 1:4])
        
        return sigma, color
```

### 4. NeRF with Uncertainty
Incorporates uncertainty estimation:

```python
class UncertaintyNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nerf = NeRF()
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )
    
    def forward(self, x, d):
        # Get intermediate features
        features = self.nerf.mlp[:-1](x)
        
        # Predict density and color
        sigma, color = self.nerf(x, d)
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(features)
        
        return sigma, color, uncertainty
```

## Performance Metrics

### 1. Novel View Synthesis Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **FID**: Fréchet Inception Distance

### 2. Training Efficiency
- **Training time**: Time to convergence
- **Memory usage**: GPU memory consumption
- **Rendering speed**: Frames per second during inference

### 3. Quality Metrics
- **Geometric accuracy**: 3D reconstruction quality
- **View consistency**: Consistency across viewpoints
- **Temporal stability**: Stability in dynamic scenes

## Applications

### 1. Novel View Synthesis
- **Virtual reality**: Immersive 3D environments
- **Augmented reality**: Realistic object insertion
- **Film production**: Virtual camera movements
- **Architecture visualization**: Walkthroughs of buildings

### 2. 3D Scene Reconstruction
- **Cultural heritage**: Preserving historical sites
- **Archaeology**: Documenting excavations
- **Real estate**: Virtual property tours
- **E-commerce**: 3D product visualization

### 3. Computer Graphics
- **Rendering**: Photorealistic image synthesis
- **Animation**: Character and scene animation
- **Visual effects**: Movie and game effects
- **Simulation**: Physics-based rendering

### 4. Scientific Visualization
- **Medical imaging**: 3D medical data visualization
- **Astronomy**: Space object visualization
- **Geology**: Terrain and structure modeling
- **Biology**: Molecular and cellular visualization

## Research Frontiers

### 1. Real-Time NeRF
- **Fast rendering**: Real-time novel view synthesis
- **Efficient training**: Reduced training time
- **Mobile deployment**: On-device NeRF rendering

### 2. Dynamic NeRF
- **Temporal modeling**: Scenes that change over time
- **Motion capture**: Human motion synthesis
- **Event modeling**: Dynamic event reconstruction

### 3. Multi-Scale NeRF
- **Hierarchical representation**: Multi-resolution scene modeling
- **Adaptive sampling**: Efficient sampling strategies
- **Progressive training**: Incremental scene refinement

### 4. NeRF for Robotics
- **SLAM**: Simultaneous localization and mapping
- **Path planning**: 3D environment understanding
- **Object manipulation**: 3D object understanding

## Usage Examples

### Basic NeRF Training
```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Initialize NeRF model
model = NeRF(pos_enc_dim=10, view_enc_dim=4, hidden_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Training loop
for epoch in range(1000):
    for batch in dataloader:
        rays_o, rays_d, target_rgb = batch
        
        # Forward pass
        loss, rgb_pred = train_nerf(model, rays_o, rays_d, target_rgb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

### Novel View Synthesis
```python
def render_novel_view(model, camera_pose, H=800, W=800, focal=1200):
    # Generate rays for novel view
    rays_o, rays_d = get_rays(H, W, focal, camera_pose)
    
    # Render image
    with torch.no_grad():
        rgb, depth, opacity, _ = render_rays(model, rays_o, rays_d)
    
    # Reshape to image
    rgb = rgb.reshape(H, W, 3)
    depth = depth.reshape(H, W)
    
    return rgb, depth

# Render novel view
camera_pose = torch.eye(4)  # Identity pose
rgb, depth = render_novel_view(model, camera_pose)

# Convert to numpy for visualization
rgb_np = rgb.cpu().numpy()
depth_np = depth.cpu().numpy()
```

### Interactive NeRF Viewer
```python
class NeRFViewer:
    def __init__(self, model, H=800, W=800, focal=1200):
        self.model = model
        self.H = H
        self.W = W
        self.focal = focal
        self.camera_pose = torch.eye(4)
    
    def update_camera(self, rotation, translation):
        # Update camera pose
        R = torch.tensor(rotation)
        t = torch.tensor(translation)
        self.camera_pose[:3, :3] = R
        self.camera_pose[:3, 3] = t
    
    def render_frame(self):
        # Render current view
        rays_o, rays_d = get_rays(self.H, self.W, self.focal, self.camera_pose)
        
        with torch.no_grad():
            rgb, depth, opacity, _ = render_rays(self.model, rays_o, rays_d)
        
        return rgb.reshape(self.H, self.W, 3).cpu().numpy()

# Usage
viewer = NeRFViewer(model)
rgb = viewer.render_frame()
```

### NeRF with Custom Dataset
```python
class CustomNeRFDataset:
    def __init__(self, image_paths, camera_poses, focal_length):
        self.image_paths = image_paths
        self.camera_poses = camera_poses
        self.focal_length = focal_length
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = load_image(self.image_paths[idx])
        
        # Get camera pose
        camera_pose = self.camera_poses[idx]
        
        # Generate rays
        rays_o, rays_d = get_rays(image.shape[0], image.shape[1], 
                                 self.focal_length, camera_pose)
        
        return rays_o, rays_d, image

# Create dataset and dataloader
dataset = CustomNeRFDataset(image_paths, camera_poses, focal_length=1200)
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

# Training
for epoch in range(1000):
    for rays_o, rays_d, target_rgb in dataloader:
        loss, rgb_pred = train_nerf(model, rays_o, rays_d, target_rgb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Files in this Directory
- `nerf_model.py`: Core NeRF implementation
- `volume_rendering.py`: Volume rendering equations
- `positional_encoding.py`: High-frequency encoding
- `ray_sampling.py`: Hierarchical ray sampling
- `example_usage.py`: Working examples
