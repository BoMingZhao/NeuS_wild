import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import mcubes
from icecream import ic
import tool.sph as sph
from einops import repeat, rearrange, reduce
from torchvision.transforms import Resize

def linear2srgb(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    else:
        pow_func = np.power
        where_func = np.where

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb

def mix(x, y, a):
    return x * (1 - a) + y * a

def compute_diffuse(diffuse, F, metalic=None):
    if metalic == None:
        return (1 - F) * diffuse / np.pi
    else:
        return F * (1 - metalic[..., None]) * diffuse / np.pi

def compute_specular(albedo, roughness, normal, view, l, metalic):
    # assume no D
    N_light = l.shape[1]
    nv = torch.mul(normal, view) # =cos(n, v)   (N_rays, 3)
    nv = torch.sum(nv, dim=-1) # (N_rays, 1)
    zero = torch.zeros_like(nv)
    nv = torch.where(nv < 0., zero, nv)

    nl = torch.mul(normal, l)
    nl = torch.sum(nl, dim=-1)
    zero = torch.zeros_like(nl)
    nl = torch.where(nl < 0., zero, nl) #(N_rays, N_light)

    h = F.normalize(view + l, p=2, dim=-1)
    nh = torch.mul(normal, h)
    nh = torch.sum(nh, dim=-1)
    nh = torch.where(nh < 0., zero, nh) #(N_rays, N_light)
    nh2 = nh * nh

    hv = torch.mul(h, view)
    hv = torch.sum(hv, dim=-1)
    hv = torch.where(hv < 0., zero, hv)[..., None]

    # D
    a = roughness * roughness
    a2 = a * a
    t = nh2 * (a2 - 1.) + 1.
    t2 = t * t
    D = a2 / (np.pi * t2) #(N_rays, N_light)

    # F
    not_metalic = 0.04 * torch.ones_like(albedo)
    F0 = mix(not_metalic, albedo, metalic[..., None]) # https://blog.csdn.net/i_dovelemon/article/details/78945950?spm=1001.2014.3001.5501
    F_ = F0 + (1 - F0) * (1 - hv)**5
    # F = repeat(F, 'n1 n2 -> n1 (k n2) c', k=N_light, c=3)

    # G
    r = roughness + 1
    r2 = r * r
    k = r2 / 8
    G1 = nv / (nv * (1 - k) + k)
    G2 = nl / (nl * (1 - k) + k)
    G = G1 * G2

    specular = (D[..., None] * F_ * G[..., None]) / (4. * nv[..., None] * nl[..., None] + 0.001)

    return F_, specular, nl

def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/4
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/4
    #                      lng = -pi
    lat_step_size = np.pi * (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph.sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas

def extract_fields(bound_min, bound_max, resolution, query_func):
    device = torch.device('cuda')
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    device = torch.device('cuda')
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 transient_network,
                 embedding_a,
                 embedding_t,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 visibility_network=None,
                 brdf_network=None,
                 sky_decoder=None,
                 affine=None,
                 embedding_affine=None,
                 shadow_net=None,
                 sky_generate_net=None,
                 embedding_generate=None):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.transient_network = transient_network
        self.visibility_network = visibility_network
        self.brdf_network = brdf_network
        self.embedding_a = embedding_a
        self.embedding_t = embedding_t
        self.sky_decoder = sky_decoder
        self.affine = affine
        self.embedding_affine = embedding_affine
        self.shadow_net = shadow_net
        self.sky_generate_net = sky_generate_net
        self.embedding_generate = embedding_generate
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None, a_embedded=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape
        device = rays_o.device

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(device)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)
        a_embedding = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=n_samples)

        density, sampled_color = nerf(pts, dirs, a_embedding)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        device = torch.device('cuda')
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        # inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        # inside_sphere = torch.ones_like(inside_sphere)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).to(device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def compute_light_visibility(self, surf, normal, chunk, near=.1, perturb_overwrite=-1, cos_anneal_ratio=0.0, sample_scale=1., 
                                train_visibility=False, select_num=1, random=True, test_time=False):
        device = torch.device('cuda')
        lpix_chunk = 256
        light_h = 16
        light_w = light_h * 2
        lxyz, lareas = gen_light_xyz(light_h, light_w)
        lxyz = torch.from_numpy(lxyz).reshape([1, -1, 3]).to(surf.device) # (1, 32 * 128, 3)
        if train_visibility:
            if random:
                idx = torch.randperm(lxyz.shape[1])
                idx = idx[:select_num]
                lxyz = lxyz[:, idx, :]
            else:
                lxyz = lxyz[:, :select_num, :]
        if test_time:
            lxyz = lxyz[:, select_num, :].reshape([-1, 1, 3])
        lareas = torch.from_numpy(lareas)
        # lxyz = torch.tensor([5., 5., 5.], dtype=torch.float32, device=surf.device).reshape([1, 1, 3])
        # lxyz = rays_d.reshape([1, 1, 3]).to(surf.device)

        n_lights = lxyz.shape[1]
        lvis_hit = torch.zeros(
            (surf.shape[0], n_lights), dtype=torch.float32, device=surf.device) # (n_surf_pts, n_lights)
        for i_ in range(0, n_lights, lpix_chunk):
            lxyz_chunk = lxyz[:, i_:i_+lpix_chunk, :] #(1, chunk, 3)

            # From surface to lights
            surf2l = lxyz_chunk - surf[:, None, :] # (n_surf_pts, chunk, 3)
            far = torch.norm(surf2l) # (n_surf_pts, chunk, 3)
            surf2l = F.normalize(surf2l, p=2, dim=-1).to(torch.float32)
            surf2l_flat = torch.reshape(surf2l, [-1, 3])

            surf_rep = repeat(surf, 'n1 c -> n1 n2 c', n2=surf2l.shape[1])
            surf_flat = torch.reshape(surf_rep, (-1, 3)) # (n_surf_pts * lpix_chunk, 3)

            # Save memory by ignoring back-lit points
            lcos = torch.einsum('ijk,ik->ij', surf2l, normal) # cos(input dot normal)
            front_lit = lcos > 0 # (n_surf_pts, lpix_chunk)
            front_lit_flat = repeat(torch.reshape(front_lit, [-1]), 'n1 -> n1 3')
            surf_flat_frontlit = torch.masked_select(surf_flat, front_lit_flat).view(-1, 3)
            surf2l_flat_frontlit = torch.masked_select(surf2l_flat, front_lit_flat).view(-1, 3)
            # far = torch.masked_select(far, front_lit).view(-1, 1)
            # far = far.to(torch.float32)
            far = 2.

            rays_o = surf_flat_frontlit
            rays_d = surf2l_flat_frontlit
            batch_size = rays_o.shape[0]
            if batch_size == 0: # no cos > 0
                continue

            # get sample points
            z_vals = torch.linspace(0.0, 1.0, self.n_samples // sample_scale).to(device)
            z_vals = near + (far - near) * z_vals[None, :]
            sample_dist = 2.0 / (self.n_samples // sample_scale)
            n_samples = self.n_samples // sample_scale
            perturb = self.perturb

            if perturb_overwrite >= 0:
                perturb = perturb_overwrite
            if perturb > 0:
                t_rand = (torch.rand([batch_size, 1]) - 0.5).to(device)
                z_vals = z_vals + t_rand * 2.0 / (self.n_samples // sample_scale)
            if self.n_importance > 0:
                with torch.no_grad():
                    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                    pts = pts.reshape([-1, 3])
                    n_rays = pts.shape[0]
                    s = []
                    for i in range(0, n_rays, chunk):
                        s += [self.sdf_network.sdf(pts[i:i+chunk])]
                    sdf = torch.cat(s, 0).reshape(batch_size, self.n_samples // sample_scale)

                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample(rays_o,
                                                    rays_d,
                                                    z_vals,
                                                    sdf,
                                                    (self.n_importance // sample_scale) // self.up_sample_steps,
                                                    64 * 2**i)
                        z_vals, sdf = self.cat_z_vals(rays_o,
                                                    rays_d,
                                                    z_vals,
                                                    new_z_vals,
                                                    sdf,
                                                    last=(i + 1 == self.up_sample_steps))

                n_samples = self.n_samples // sample_scale + self.n_importance // sample_scale

            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(device)], -1)
            mid_z_vals = z_vals + dists * 0.5

            pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
            dirs = rays_d[:, None, :].expand(pts.shape)
            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)

            # get sdf and gradients
            n_rays = pts.shape[0]
            s = []
            for i in range(0, n_rays, chunk):
                s += [self.sdf_network(pts[i:i+chunk])]
            sdf_nn_output = torch.cat(s, 0)
            sdf = sdf_nn_output[:, :1]
            with torch.enable_grad():
                n_rays = pts.shape[0]
                g = []
                for i in range(0, n_rays, chunk):
                    g += [self.sdf_network.gradient(pts[i:i+chunk]).squeeze()]
            gradients = torch.cat(g, 0)

            inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
            inv_s = inv_s.expand(batch_size * n_samples, 1)

            true_cos = (dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
            transmittance = torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

            occu = transmittance[:, -1].squeeze()

            front_lit_full = torch.zeros_like(lvis_hit, dtype=bool)
            front_lit_full[:, i_:i_+lpix_chunk] = front_lit
            lvis_hit[front_lit_full] = 1 - occu
        if train_visibility and random:
            view = lxyz - surf[:, None, :]
            view = F.normalize(view, p=2, dim=-1).to(torch.float32)
            return lvis_hit, view
        return lvis_hit

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    transient_network,
                    a_embedded,
                    t_embedded,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    test_time=False,
                    use_transient=True,
                    compute_visibility=False,
                    train_visibility=False,
                    visibility_net=None,
                    relit=False,
                    envmap=None,
                    affine_embedded=None,
                    generate_embedded=None):
        batch_size, n_samples = z_vals.shape
        device = torch.device('cuda')
        # Section length
        with torch.no_grad():
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(device)], -1)
            mid_z_vals = z_vals + dists * 0.5

            # Section midpoints
            pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
            dirs = rays_d[:, None, :].expand(pts.shape)

            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)

            sdf_nn_output = sdf_network(pts)
            sdf = sdf_nn_output[:, :1]
            feature_vector = sdf_nn_output[:, 1:]
        a_embedding = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=n_samples)
        t_embedding = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=n_samples)
        feature_vector_a = torch.cat([feature_vector, a_embedding], dim=-1)
        feature_vector_t = torch.cat([feature_vector, t_embedding], dim=-1)

        with torch.enable_grad():
            gradients = sdf_network.gradient(pts).squeeze()
        with torch.no_grad():
            sampled_color = color_network(pts, gradients, dirs, feature_vector_a).reshape(batch_size, n_samples, 3)
            # sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)
            if not test_time and not relit and not train_visibility and use_transient:
                transient_out = transient_network(feature_vector_t).reshape(batch_size, n_samples, -1)
                transient_rgbs = transient_out[..., :3]
                transient_sigmas = transient_out[..., 3]
                transient_betas = transient_out[..., 4]

                deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
                delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
                deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)


                transient_alphas = 1-torch.exp(-deltas * transient_sigmas)

                alphas_shifted = \
                        torch.cat([torch.ones_like(transient_alphas[:, :1]), 1-transient_alphas], -1) # [1, 1-a1, 1-a2, ...]
                transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
                transient_weights = transient_alphas * transmittance

                transient_color = rearrange(transient_weights, 'n1 n2 -> n1 n2 1') * transient_rgbs
                sampled_color = sampled_color + transient_color

            inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
            inv_s = inv_s.expand(batch_size * n_samples, 1)

            true_cos = (dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

            pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
            # inside_sphere = (pts_norm < 1.0).float().detach()
            inside_sphere = (pts_norm < 1.0).float().detach()
            # inside_sphere = torch.ones_like(inside_sphere).float().detach()
            relax_inside_sphere = (pts_norm < 1.2).float().detach()
            # relax_inside_sphere = torch.ones_like(pts_norm).float().detach()

            # Render with background
            if background_alpha is not None:
                alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
                alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
                sampled_color = sampled_color * inside_sphere[:, :, None] +\
                                background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
                sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
            weights_sum = weights.sum(dim=-1, keepdim=True)
            depth = reduce(weights * mid_z_vals, 'n1 n2 -> n1', 'sum')
            surface = rays_o.squeeze() + rays_d.squeeze() * depth[:, None]
            sdf_ = self.sdf_network(surface)[:, :1]

        if compute_visibility:
            with torch.enable_grad():
                normal = self.sdf_network.gradient(surface).squeeze()
            visibility = self.compute_light_visibility(surface, normal, 1024, cos_anneal_ratio=cos_anneal_ratio, sample_scale=4)
            results = {
                'depth': depth,
                'visibility': visibility
            }
            return results
        if train_visibility:
            if test_time:
                lxyz, _ = gen_light_xyz(16, 32)
                lxyz = torch.from_numpy(lxyz).reshape([1, -1, 3]).to(surface.device) # (1, 32 * 128, 3)
                view = lxyz - surface[:, None, :]
                view = F.normalize(view, p=2, dim=-1).to(torch.float32)
                n_lights = view.shape[1]
                batch_size = view.shape[0]

                surface_input = repeat(surface, 'n1 n2 -> (n1 n3) n2', n3=n_lights) # (batch * sample_light, 3)
                view = view.reshape([-1, 3])
                v_pred = visibility_net(surface_input, view).reshape([batch_size, n_lights])
                v_gt = None
                '''with torch.enable_grad():
                    normal = self.sdf_network.gradient(surface).squeeze()
                v_gt = self.compute_light_visibility(surface, normal, 4 * 1024, cos_anneal_ratio=cos_anneal_ratio, sample_scale=2, 
                                                     train_visibility=False, select_num=64, test_time=True)'''
                results = {
                    'v_gt': v_gt,
                    'v_pred': v_pred
                }

                return results
                
            '''with torch.enable_grad():
                normal = self.sdf_network.gradient(surface).squeeze()
            with torch.no_grad():
                normal = F.normalize(normal, p=2, dim=-1).to(torch.float32)
                chunk = batch_size * 2
                n_points = pts.shape[0]
                v_gt_chunk = []
                view_chunk = []
                for i in range(0, n_points, chunk):
                    v_gt_, view_ = self.compute_light_visibility(pts[i:i+chunk], normal[i:i+chunk], 4 * 1024, cos_anneal_ratio=cos_anneal_ratio, sample_scale=2, 
                                                        train_visibility=True, select_num=64, random=True)
                    v_gt_chunk.append(v_gt_)
                    view_chunk.append(view_)
                v_gt = torch.cat(v_gt_chunk, dim=0)
                view = torch.cat(view_chunk, dim=0)'''
            
            with torch.no_grad():
                noise = torch.normal(1, 0, size=(batch_size, 3)).to(surface.device)
                surface_input = surface + noise
                with torch.enable_grad():
                    normal = self.sdf_network.gradient(surface_input).squeeze()
                normal = F.normalize(normal, p=2, dim=-1).to(torch.float32)
                v_gt, view = self.compute_light_visibility(surface_input, normal, 4 * 1024, cos_anneal_ratio=cos_anneal_ratio, sample_scale=2, 
                                                            train_visibility=True, select_num=100, random=True)
                n_lights = view.shape[1]
                
                surface_v = repeat(surface_input, 'n1 n2 -> (n1 n3) n2', n3=n_lights) # (batch * sample_light, 3)
                view = view.reshape([-1, 3])
                
            v_pred = visibility_net(surface_v, view).reshape([-1, n_lights])

            results = {
                'v_gt': v_gt,
                'v_pred': v_pred
            }
            return results

        if relit: # relit network
            # compute light view and normal
            with torch.enable_grad():
                # normal = self.sdf_network.gradient(surface).squeeze()
                normal = self.sdf_network.gradient(pts).squeeze()
                # normal = repeat(normal, 'n1 c -> (n1 n2) c', n2=n_samples)
            v_surface = None
            normal = F.normalize(normal, p=2, dim=-1).to(torch.float32)
            lxyz, larea = gen_light_xyz(16, 32)
            larea = torch.from_numpy(larea).reshape([1, -1, 1]).to(surface.device)
            lxyz = torch.from_numpy(lxyz).reshape([1, -1, 3]).to(surface.device) # (1, 32 * 128, 3)
            # view = lxyz - surface[:, None, :]
            view = repeat(lxyz, 'n1 n2 c -> (n1 b) n2 c', b=pts.shape[0]) # (batch * n_samples, 512, 3)
            view = F.normalize(view, p=2, dim=-1).to(torch.float32)
            n_lights = view.shape[1]
            n_rays = view.shape[0]

            # get visibility
            # surface_input = repeat(surface, 'n1 n2 -> (n1 n3) n2', n3=n_lights) # (batch * sample_light, 3)
            pts_input = repeat(pts, 'n1 n2 -> (n1 n3) n2', n3=n_lights)# (batch * sample_light * n_samples, 3)
            view = view.reshape([-1, 3])
            B = pts_input.shape[0]            
            if self.shadow_net != None:
                v_pred = 1.
                shade = self.shadow_net(pts, a_embedding).reshape([batch_size, n_samples])
                v_surface = reduce(weights * shade, 'n1 n2 -> n1', 'sum')
                # v_surface = repeat(v_surface, 'n1 n2 -> n1 n2 c', c=3)
            else:
                chunk = batch_size * n_lights * 4
                '''v_pred_chunk = []
                for i in range(0, B, chunk):
                    v_pred_chunk += [visibility_net(pts_input[i:i+chunk], view[i:i+chunk])]
                v_pred = torch.cat(v_pred_chunk, dim=0).reshape([n_rays, n_lights])'''
                # v_pred1 = torch.ones_like(v_pred)
                # v_pred1 = repeat(v_pred1, 'n1 n2 -> (n1 n3) n2', n3=n_samples) # (batch * n_samples. n_lights)
            
            
            # get brdf
            # brdf = self.brdf_network(surface, a_embedded) # [batch, 3]
            brdf = self.brdf_network(pts) # [batch * n_samples, 3]
            if self.shadow_net != None:
                shade = shade.reshape([-1, 1])
                # shade1 = torch.ones_like(shade)
                albedo = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * brdf[..., :3].reshape([batch_size, n_samples, 3]),
                             'n1 n2 c -> n1 c', 'sum')
                brdf = brdf * shade
            else:
                albedo = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * brdf[..., :3].reshape([batch_size, n_samples, 3]),
                                'n1 n2 c -> n1 c', 'sum')

            # get env map
            if envmap != None:
                light = repeat(envmap, 'n2 c -> n1 n2 c', n1 = batch_size)
                v_light = (v_pred[..., None] * light).sum(dim=-1)[:, 255]
                v_light = torch.clip(v_light, 0., 1.).reshape([-1])
                v_gt = self.compute_light_visibility(surface, normal, 1024, cos_anneal_ratio=cos_anneal_ratio, sample_scale=4, select_num=255, test_time=True)
            else:
                hdr = self.sky_decoder(a_embedded)
                hdr.clamp_min_(0.0)
                torch_resize = Resize([16, 32])
                hdr_resize = torch_resize(hdr)
                hdr_resize = torch.permute(hdr_resize, (0, 2, 3, 1))
                light = hdr_resize.view([batch_size, -1, 3]) # need resize
                light = repeat(light, 'n1 n2 c -> (n1 n3) n2 c', n3=n_samples)
                       
            '''a_embedding = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=n_lights)
            hdr = self.sky_decoder(view, a_embedding)
            light = hdr.reshape([batch_size, n_lights, 3])
            view = view.reshape([batch_size, n_lights, 3])
            l_out = F.normalize(-1 * rays_d.squeeze(), p=2, dim=-1)'''

            view = view.reshape([-1, n_lights, 3])
            # view = repeat(view, 'n1 n2 c -> (n1 n3) n2 c', n3=n_samples) # (batch * n_samples, n_lights, 3)

            # compute brdf color
            if self.shadow_net != None:
                static_rgb_map = self.brdf_render(brdf[:, None ,:3], None, normal[:, None, :], None, 
                                              view, light, 1., None, None) #(batch, 3)
            else:
                static_rgb_map = self.brdf_render(brdf[:, None ,:3], None, normal[:, None, :], None, 
                                              view, light, 1., None, None) #(batch, 3)
                '''static_rgb_map = self.brdf_render(brdf[:, None ,:3], None, normal[:, None, :], None, 
                                                view, light, v_pred[..., None], None, None) #(batch, 3)'''
            static_rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * static_rgb_map.reshape([batch_size, n_samples, 3]),
                             'n1 n2 c -> n1 c', 'sum')

            if self.affine != None: # use affine
                affine_matrix = self.affine(affine_embedded).reshape([batch_size, -1])
                if affine_matrix.shape[1] == 12: # use 3*4 affine matrix
                    affine_matrix = self.affine(affine_embedded).reshape([batch_size, 3, 4]) # (batch, 3*4)
                    transpose = affine_matrix[:, :, :3]
                    transpose_eye = repeat(torch.eye(3), 'c1 c2 -> n c1 c2', n=transpose.shape[0]).to(transpose.device)
                    transpose = transpose + transpose_eye
                    offset = affine_matrix[:, :, 3:4].reshape([batch_size, 3])
                    static_rgb_map = torch.bmm(static_rgb_map[:, None, :], transpose).reshape([batch_size, 3]) + offset
                else:
                    affine_matrix = self.affine(affine_embedded).reshape([batch_size, 6])
                    scale = affine_matrix[:, 0:3].reshape([batch_size, 3])
                    offset = affine_matrix[:, 3:6].reshape([batch_size, 3])
                    static_rgb_map = static_rgb_map * scale + offset

            if self.sky_generate_net != None:
                sky = self.sky_generate_net(rays_d, generate_embedded)
                sky_rgb = sky * (1 - weights_sum)
                static_rgb_map = static_rgb_map + sky_rgb

            static_rgb_map = torch.clip(static_rgb_map, 0., 1.)
            albedo = torch.clip(albedo, 0., 1.)
            final_color = static_rgb_map
            # compute transient color
            if not test_time and use_transient:
                feature_t = torch.cat([pts, t_embedding], dim=-1)
                transient_out = transient_network(feature_t).reshape([batch_size, n_samples, -1]) #[batch_size, 5]
                transient_rgbs = transient_out[..., :3]
                transient_sigmas = transient_out[..., 3]
                transient_betas = transient_out[..., 4]

                deltas = mid_z_vals[:, 1:] - mid_z_vals[:, :-1] # (N_rays, N_samples_-1)
                delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
                deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

                transient_alphas = 1-torch.exp(-deltas * transient_sigmas)

                alphas_shifted = \
                        torch.cat([torch.ones_like(transient_alphas[:, :1]), 1-transient_alphas], -1) # [1, 1-a1, 1-a2, ...]
                transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
                transient_weights = transient_alphas * transmittance

                transient_rgb_map = reduce(rearrange(transient_weights, 'n1 n2 -> n1 n2 1') * transient_rgbs, 'n1 n2 c -> n1 c', 'sum')
                final_color = final_color + transient_rgb_map

                betas = reduce(transient_weights * transient_betas, 'n1 n2 -> n1', 'sum')
                betas += transient_network.beta_min
            
            results = {
                'color': final_color, 
                'color_static': static_rgb_map,
                'albedo': albedo,
                'sdf': sdf_
            }
            
            if envmap != None:
                results['v'] = v_light
                results['v_gt'] = v_gt
            
            if not test_time and use_transient:
                results['betas'] = betas
                results['transient_sigmas'] = transient_sigmas

            if self.affine != None:
                results['affine'] = affine_matrix.reshape([batch_size, -1])

            if v_surface != None:
                results['v_surface'] = v_surface
                results['shade'] = shade

            return results

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        if not test_time:
            betas = reduce(transient_weights * transient_betas, 'n1 n2 -> n1', 'sum')
            betas += transient_network.beta_min

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        results = {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'depth': depth
        }
        if not test_time:
            results['betas'] = betas
            results['transient_sigmas'] = transient_sigmas
        return results

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, ts=None, 
               test_time=False, use_transient=True, compute_visibility=False, train_visibility=False, relit=False, envmap=None):
        device = torch.device('cuda')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside).to(device)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).to(device)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]]).to(device)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        a_embedded = self.embedding_a(ts)
        t_embedded = self.embedding_t(ts)
        if self.embedding_affine != None:
            affine_embedded = self.embedding_affine(ts)
        else:
            affine_embedded = None

        if self.embedding_generate != None:
            generate_embedded = self.embedding_generate(ts)
        else:
            generate_embedded = None
        
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf, a_embedded=a_embedded)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    self.transient_network,
                                    a_embedded,
                                    t_embedded,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    test_time=test_time,
                                    use_transient=use_transient,
                                    compute_visibility=compute_visibility,
                                    train_visibility=train_visibility,
                                    visibility_net=self.visibility_network,
                                    relit=relit,
                                    envmap=envmap,
                                    affine_embedded=affine_embedded,
                                    generate_embedded=generate_embedded)
        if compute_visibility:
            results = {
                'depth': ret_fine['depth'],
                'visibility': ret_fine['visibility']
            }
            
            return results

        if train_visibility:
            results = {
                'v_pred': ret_fine['v_pred'],
                'v_gt': ret_fine['v_gt']
            }
            
            return results

        if relit:
            results = {
                'color_fine': ret_fine['color'],
                'color_static': ret_fine['color_static'],
                'albedo': ret_fine['albedo'],
                'sdf': ret_fine['sdf']
            }

            if 'v_gt' in ret_fine:
                results['v'] = ret_fine['v']
                results['v_gt'] = ret_fine['v_gt']

            if 'transient_sigmas' in ret_fine:
                results['betas'] = ret_fine['betas']
                results['transient_sigmas'] = ret_fine['transient_sigmas']

            if self.affine != None:
                results['affine'] = ret_fine['affine']
            
            if 'v_surface' in ret_fine:
                results['v_surface'] = ret_fine['v_surface']
                results['shade'] = ret_fine['shade']

            return results

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        results = {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'depth': ret_fine['depth']
        }

        if 'transient_sigmas' in ret_fine:
            results['betas'] = ret_fine['betas']
            results['transient_sigmas'] = ret_fine['transient_sigmas']

        return results

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))

    def brdf_render(self, diffuse, roughness, normal, view, l, light, visibility=1., metalic=None, area=None): # For now, we just use lambert brdf
        nl = torch.mul(normal, l)
        nl = torch.sum(nl, dim=-1)
        zero = torch.zeros_like(nl)
        nl = torch.where(nl < 0., zero, nl) #(N_rays, N_light)

        '''F, spec, nl = compute_specular(diffuse, roughness, normal, view, l, metalic)
        diff = compute_diffuse(diffuse, F, metalic)
        rgb = torch.sum((diff + spec) * light * visibility * nl[..., None], dim=1)'''
        diff = compute_diffuse(diffuse, 0.)
        if area == None:
            rgb = torch.sum(diff * light * visibility * nl[..., None], dim=1)
        else:
            rgb = torch.sum(diff * light * visibility * nl[..., None] * area, dim=1)
        
        rgb = torch.clip(rgb, 0.001, 1.)

        rgb = linear2srgb(rgb)
        # rgb = rgb / (1. + rgb)

        return rgb

    