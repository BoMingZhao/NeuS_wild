import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 a_in=64,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False,
                 use_a=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None
        self.a_in = a_in
        self.use_a=use_a

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view  + self.a_in + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views, input_a=None):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            if self.use_a:
                h = torch.cat([feature, input_views, input_a], -1)
            else:
                h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).to(self.variance.device) * torch.exp(self.variance * 10.0)


class NeRF_transient(nn.Module):
    def __init__(self, W=256,
                 in_channels_xyz=63, in_channels_t=16,
                 beta_min=0.03, use_feature=True, multires=10., d_in=3.):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t
        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.W = W
        self.in_channels_xyz = in_channels_xyz

        self.in_channels_t = in_channels_t
        self.beta_min = beta_min
        self.use_feature = use_feature
        self.d_in = d_in

        if not use_feature:
            embed_fn, input_ch_xyz = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            self.input_ch_xyz = input_ch_xyz

            self.xyz_encoding_final = nn.Linear(self.input_ch_xyz, W)

        # xyz encoding layers
        else:
            self.xyz_encoding_final = nn.Linear(in_channels_xyz, W)

        self.transient_encoding = nn.Sequential(
                                    nn.Linear(W + in_channels_t, W//2), nn.ReLU(True),
                                    nn.Linear(W//2, W//2), nn.ReLU(True),
                                    nn.Linear(W//2, W//2), nn.ReLU(True),
                                    nn.Linear(W//2, W//2), nn.ReLU(True))
            


        # transient output layers
        self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
        self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
        self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py
        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.
        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if self.use_feature:
            input_xyz, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_t = \
                torch.split(x, [self.d_in,
                                self.in_channels_t], dim=-1)
            input_xyz = self.embed_fn_fine(input_xyz)
        xyz_ = input_xyz
        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)
        return transient

class NeRF_visibility(nn.Module):
    def __init__(self, 
                 D=4, W=128, skips=[2], multires=10., multires_view=4., d_in=3, d_out=512):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips

        embed_fn, input_ch_xyz = get_embedder(multires, input_dims=d_in)
        self.embed_fn_fine = embed_fn
        self.input_ch_xyz = input_ch_xyz

        embedview_fn, input_ch_view = get_embedder(multires_view)
        self.embedview_fn = embedview_fn
        self.input_ch_view = input_ch_view

        self.in_channels_xyz = input_ch_xyz + input_ch_view

        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU())
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Sequential(nn.Linear(W, d_out), nn.Sigmoid())

    def forward(self, x, view):
        input_xyz = self.embed_fn_fine(x)
        input_view = self.embedview_fn(view)
        xyz_ = torch.cat([input_xyz, input_view], 1)
        input_xyz = xyz_
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        out = self.xyz_encoding_final(xyz_)

        return out

class shadow(nn.Module):
    def __init__(self, 
                 D=4, W=256, skips=[2], multires=10., embedding_in=64, d_in=3, d_out=1):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips

        embed_fn, input_ch_xyz = get_embedder(multires, input_dims=d_in)
        self.embed_fn_fine = embed_fn
        self.input_ch_xyz = input_ch_xyz

        self.in_channels_xyz = input_ch_xyz + embedding_in

        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU())
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Sequential(nn.Linear(W, d_out), nn.Sigmoid())

    def forward(self, x, embedding):
        input_xyz = self.embed_fn_fine(x)
        xyz_ = torch.cat([input_xyz, embedding], 1)
        input_xyz = xyz_
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        out = self.xyz_encoding_final(xyz_)

        return out

class Brdf(nn.Module):
    def __init__(self, 
                 D=4, W=256, skips=[2], multires=10., d_in=3, d_out=3, d_embedding=0):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.d_embedding = d_embedding

        embed_fn, input_ch_xyz = get_embedder(multires, input_dims=d_in)
        self.embed_fn_fine = embed_fn
        self.input_ch_xyz = input_ch_xyz + self.d_embedding

        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.input_ch_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + self.input_ch_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU())
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Sequential(nn.Linear(W, d_out), nn.Sigmoid())

    def forward(self, x, embedding=None):
        input_xyz = self.embed_fn_fine(x)
        if self.d_embedding != 0:
            input_xyz = torch.cat([input_xyz, embedding], dim=-1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        out = self.xyz_encoding_final(xyz_)
        # out = torch.cat([out, xyz_], -1)

        return out

class Sky(nn.Module):
    def __init__(self, 
                 D=4, W=256, skips=[2], multires_view=4, d_in=3, d_out=3):
        super().__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.skips = skips

        embedview_fn, input_ch_view = get_embedder(multires_view)
        self.embedview_fn = embedview_fn
        self.input_ch_view = input_ch_view


        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.d_in + self.input_ch_view, W)
            elif i in skips:
                layer = nn.Linear(W + self.d_in + self.input_ch_view, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU())
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Sequential(nn.Linear(W, d_out), nn.ReLU())

    def forward(self, view, feature):
        view_input = self.embedview_fn(view)
        input_all = torch.cat([view_input, feature], dim=-1)
        xyz_ = input_all
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([xyz_, input_all], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        out = self.xyz_encoding_final(xyz_)
        # out = torch.cat([out, xyz_], -1)

        return out

class affine_net(nn.Module):
    def __init__(self, 
                 D=4, W=256, skips=[2], d_in=64, d_scale=1, d_offset=1, use_matrix=True):
        super().__init__()

        self.D = D
        self.W = W
        self.skips = skips
        self.d_in = d_in
        self.use_matrix = use_matrix

        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.d_in, W)
            elif i in skips:
                layer = nn.Linear(W + self.d_in , W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU())
            setattr(self, f"embedding_encoding_{i+1}", layer)
        
        if not use_matrix:
            layer_scale  = nn.Linear(W, d_scale)
            layer_offset = nn.Linear(W, d_offset)
            nn.init.constant(layer_scale.bias, 1)
            nn.init.constant(layer_scale.weight, 0)
            nn.init.constant(layer_offset.weight, 0)
            nn.init.constant(layer_offset.bias, 0)

            self.embedding_encoding_scale = nn.Sequential(layer_scale)
            self.embedding_encoding_offset = nn.Sequential(layer_offset)
        else:
            layer_scale  = nn.Linear(W, d_scale)
            self.embedding_encoding_scale = nn.Sequential(layer_scale)

    def forward(self, embedding):
        input_embedding = embedding

        for i in range(self.D):
            if i in self.skips:
                input_embedding = torch.cat([input_embedding, embedding], -1)
            input_embedding = getattr(self, f"embedding_encoding_{i+1}")(input_embedding)
        if not self.use_matrix:
            out_scale = self.embedding_encoding_scale(input_embedding)
            out_offset = self.embedding_encoding_offset(input_embedding)
            out = torch.cat([out_scale, out_offset], dim=-1)
        else:
            out = self.embedding_encoding_scale(input_embedding)

        return out

class sky_generate(nn.Module):
    def __init__(self, 
                 D=3, W=256, multires_view=4, d_in=3, d_out=3, d_embedding=64):
        super().__init__()
        self.D = D
        self.W = W
        self.d_embedding = d_embedding

        embed_fn, input_ch_view = get_embedder(multires_view, input_dims=d_in)
        self.embed_fn_fine = embed_fn
        self.input_ch = input_ch_view + self.d_embedding

        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.input_ch, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU())
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Sequential(nn.Linear(W, d_out), nn.Sigmoid())

    def forward(self, view, embedding):
        input_view = self.embed_fn_fine(view)
        input_view = torch.cat([input_view, embedding], dim=-1)
        xyz_ = input_view
        for i in range(self.D):
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        out = self.xyz_encoding_final(xyz_)

        return out
