import numpy as np

def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.
    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    # _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    '''z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)'''
    y = -1 * r * np.sin(lat)
    z = -1 * r * np.cos(lat) * np.cos(lng)
    x = -1 * r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart
