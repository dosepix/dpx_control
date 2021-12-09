"""
Rebin 1D and 2D histograms.

"""

import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt

try:
    import uncertainties.unumpy as unp 
    nom = unp.nominal_values
except ImportError:
    nom = lambda x: x

from bounded_splines import BoundedUnivariateSpline, BoundedRectBivariateSpline

def midpoints(xx):
    """Return midpoints of edges in xx."""
    return xx[:-1] + 0.5*np.ediff1d(xx)


def edge_step(x, y, **kwargs):
    """
    Plot a histogram with edges and bin values precomputed. The normal
    matplotlib hist function computes the bin values internally.

    Input
    -----
     * x : n+1 array of bin edges.
     * y : n array of histogram values.

    """
    return plt.plot(x, np.hstack([y, y[-1]]), drawstyle='steps-post', **kwargs)


def rebin_along_axis(y1, x1, x2, axis=0, interp_kind=3):
    """
    Rebins an N-dimensional array along a given axis, in a piecewise-constant
    fashion.

    Parameters
    ----------
    y1 : array_like
        The input image
    x1 : array_like
        The monotonically increasing/decreasing original bin edges along
        `axis`, must be 1 greater than `np.size(y1, axis)`.
    y2 : array_like
        The final bin_edges along `axis`.
    axis : int
        The axis to be rebinned, it must exist in the original image.
    interp_kind : how is the underlying unknown continuous distribution
                  assumed to look: {3, 'piecewise_constant'}
                  3 is cubic splines
                  piecewise_constant is constant in each histogram bin

    Returns
    -------
    output : np.ndarray
        The rebinned image.
    """

    orig_shape = np.array(y1.shape)
    num_axes = np.size(orig_shape)

    # Output is going to need reshaping
    new_shape = np.copy(orig_shape)
    new_shape[axis] = np.size(x2) - 1

    if axis > num_axes - 1:
        raise ValueError("That axis is not in y1")

    if np.size(y1, axis) != np.size(x1) - 1:
        raise ValueError("The original number of xbins does not match the axis"
                         "size")

    odtype = np.dtype('float')
    if y1.dtype is np.dtype('O'):
        odtype = np.dtype('O')

    output = np.empty(new_shape, dtype=odtype)

    it = np.nditer(y1, flags=['multi_index', 'refs_ok'])
    it.remove_axis(axis)

    while not it.finished:
        a = list(it.multi_index)
        a.insert(axis, slice(None))

        rebinned = rebin(x1, y1[a], x2, interp_kind=interp_kind)

        output[a] = rebinned[:]
        it.iternext()

    return output


def rebin(x1, y1, x2, interp_kind=3):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.

    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in 
              each bin, not an average.
     * x2 : n+1 array of new bin edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {3, 'piecewise_constant'}
                      3 is cubic splines
                      piecewise_constant is constant in each histogram bin


    Returns
    -------
     * y2 : n array of rebinned histogram values.

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """

    if interp_kind == 'piecewise_constant':
        return rebin_piecewise_constant(x1, y1, x2)
    else:
        return rebin_spline(x1, y1, x2, interp_kind=interp_kind)
     

def rebin_spline(x1, y1, x2, interp_kind):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.

    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in 
              each bin, not an average.
     * x2 : n+1 array of new bin edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {'cubic'}

    Returns
    -------
     * y2 : n array of rebinned histogram values.

    The cubic spline fit (which is the only interp_kind tested) 
    uses the UnivariateSpline class from Scipy, which uses FITPACK.
    The boundary condition used is not-a-knot, where the second and 
    second-to-last nodes are not included as knots (but they are still
    interpolated).

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """
    m = y1.size
    n = x2.size - 1

    # midpoints of x1
    x1_mid = midpoints(x1)

    # constructing data for spline
    #  To get the spline to flatten out at the edges, duplicate bin mid values
    #   as value on the two boundaries.
    xx = np.hstack([x1[0], x1_mid, x1[-1]])
    yy = np.hstack([y1[0], y1, y1[-1]])

    # strip uncertainties from data
    yy = nom(yy)

    # instantiate spline, s=0 gives interpolating spline
    spline = BoundedUnivariateSpline(xx, yy, s=0., k=interp_kind)

    # area under spline for each old bin
    areas1 = np.array([spline.integral(x1[i], x1[i+1]) for i in range(m)])


    # insert old bin edges into new edges
    x1_in_x2 = x1[ np.logical_and(x1 > x2[0], x1 < x2[-1]) ]
    indices  = np.searchsorted(x2, x1_in_x2)
    subbin_edges = np.insert(x2, indices, x1_in_x2)

    # integrate over each subbin
    subbin_areas = np.array([spline.integral(subbin_edges[i], 
                                             subbin_edges[i+1]) 
                              for i in range(subbin_edges.size-1)])

    # make subbin-to-old bin map
    subbin_mid = midpoints(subbin_edges)
    sub2old = np.searchsorted(x1, subbin_mid) - 1

    # make subbin-to-new bin map
    sub2new = np.searchsorted(x2, subbin_mid) - 1

    # loop over subbins
    y2 = [0. for i in range(n)]
    for i in range(subbin_mid.size):
        # skip subcells which don't lie in range of x1
        if sub2old[i] == -1 or sub2old[i] == x1.size-1:
            continue
        else:
            y2[sub2new[i]] += ( y1[sub2old[i]] * subbin_areas[i] 
                                               / areas1[sub2old[i]] )

    return np.array(y2)


def rebin_piecewise_constant(x1, y1, x2):
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)

    # the fractional bin locations of the new bins in the old bins
    i_place = np.interp(x2, x1, np.arange(len(x1)))

    cum_sum = np.r_[[0], np.cumsum(y1)]

    # calculate bins where lower and upper bin edges span
    # greater than or equal to one original bin.
    # This is the contribution from the 'intact' bins (not including the
    # fractional start and end parts.
    whole_bins = np.floor(i_place[1:]) - np.ceil(i_place[:-1]) >= 1.
    start = cum_sum[np.ceil(i_place[:-1]).astype(int)]
    finish = cum_sum[np.floor(i_place[1:]).astype(int)]

    y2 = np.where(whole_bins, finish - start, 0.)

    bin_loc = np.clip(np.floor(i_place).astype(int), 0, len(y1) - 1)

    # fractional contribution for bins where the new bin edges are in the same
    # original bin.
    same_cell = np.floor(i_place[1:]) == np.floor(i_place[:-1])
    frac = i_place[1:] - i_place[:-1]
    contrib = (frac * y1[bin_loc[:-1]])
    y2 += np.where(same_cell, contrib, 0.)

    # fractional contribution for bins where the left and right bin edges are in
    # different original bins.
    different_cell = np.floor(i_place[1:]) > np.floor(i_place[:-1])
    frac_left = np.ceil(i_place[:-1]) - i_place[:-1]
    contrib = (frac_left * y1[bin_loc[:-1]])

    frac_right = i_place[1:] - np.floor(i_place[1:])
    contrib += (frac_right * y1[bin_loc[1:]])

    y2 += np.where(different_cell, contrib, 0.)

    return y2


def rebin2d(x1, y1, z1, x2, y2, interp_kind=3):
    """
    Rebin 2d histogram values z1 from old rectangular bin 
    edges x1, y1 to new edges x2, y2.

    Input
    -----
     * x1 : m+1 array of old bin x edges.
     * y1 : n+1 array of old bin y edges.
     * z1 : m-by-n array of old histogram values. This is the total number in 
              each bin, not an average.
     * x2 : p+1 array of new bin x edges.
     * x2 : q+1 array of new bin y edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {3}
                      3 - bivariate cubic spline

    Returns
    -------
     * z2 : p-by-q array of rebinned histogram values.

    The cubic spline fit (which is the only interp_kind tested) 
    uses the BivariateSpline class from Scipy, which uses FITPACK.
    The boundary condition used is not-a-knot, where the second and 
    second-to-last nodes are not included as knots (but they are still
    interpolated).

    Bins in x2 x y2 that are entirely outside the range of x1 x y1 
    are assigned 0.
    """
    m, n = z1.shape
    assert x1.size == m+1
    assert y1.size == n+1

    p = x2.size - 1
    q = y2.size - 1

    # midpoints of x1
    x1_mid = midpoints(x1)
    y1_mid = midpoints(y1)

    # constructing data for spline
    #  To get the spline to flatten out at the edges, duplicate bin mid values
    #   on the interpolation boundaries.
    xx = np.hstack([x1[0], x1_mid, x1[-1]])
    yy = np.hstack([y1[0], y1_mid, y1[-1]])

    c1 = np.hstack([z1[0,0], z1[:,0], z1[-1,0]])
    c2 = np.vstack([z1[0,:], z1, z1[-1,:]])
    c3 = np.hstack([z1[0,-1], z1[:,-1], z1[-1,-1]])

    zz = np.hstack([c1[:,np.newaxis], c2, c3[:,np.newaxis]])
    
    zz = nom(zz)

    # instantiate spline, s=0 gives interpolating spline
    spline = BoundedRectBivariateSpline(xx, yy, zz, s=0., 
                                        kx=interp_kind,
                                        ky=interp_kind)

    # area under spline for each old bin
    # todo: only integrate over old bins which will contribute to new bins
    areas1 = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            areas1[i,j] = spline.integral(x1[i], x1[i+1], y1[j], y1[j+1])


    # insert old bin edges into new edges
    #  into x
    x1_in_x2 = x1[ np.logical_and(x1 > x2[0], x1 < x2[-1]) ]
    x_indices  = np.searchsorted(x2, x1_in_x2)
    subbin_xedges = np.insert(x2, x_indices, x1_in_x2)

    #  into y
    y1_in_y2 = y1[ np.logical_and(y1 > y2[0], y1 < y2[-1]) ]
    y_indices  = np.searchsorted(y2, y1_in_y2)
    subbin_yedges = np.insert(y2, y_indices, y1_in_y2)

    # integrate over each subbin
    ms = subbin_xedges.size-1
    ns = subbin_yedges.size-1
    subbin_areas = np.zeros((ms,ns))
    for i in range(ms):
        for j in range(ns):
            subbin_areas[i,j] = spline.integral(
                                    subbin_xedges[i], subbin_xedges[i+1],
                                    subbin_yedges[j], subbin_yedges[j+1],
                                               )


    # make subbin-to-old bin map
    subbin_xmid = midpoints(subbin_xedges)
    x_sub2old = np.searchsorted(x1, subbin_xmid) - 1

    subbin_ymid = midpoints(subbin_yedges)
    y_sub2old = np.searchsorted(y1, subbin_ymid) - 1

    # make subbin-to-new bin map
    x_sub2new = np.searchsorted(x2, subbin_xmid) - 1
    y_sub2new = np.searchsorted(y2, subbin_ymid) - 1

    # loop over subbins
    z2 = [[0. for i in range(q)] for j in range(p)]
    for i in range(ms):
        for j in range(ns):
            # skip subcells which don't lie in range of x1 or y1
            if ( x_sub2old[i] == -1 or x_sub2old[i] == m or
                 y_sub2old[j] == -1 or y_sub2old[j] == n ):
                continue
            else:
                z2[x_sub2new[i]][y_sub2new[j]] += ( 
                         z1[x_sub2old[i],y_sub2old[j]] 
                         * subbin_areas[i,j] / 
                         areas1[x_sub2old[i], y_sub2old[j]] )

    return np.array(z2)

if __name__ == '__main__':
    # demo rebin() ---------------------------------------------------

    # old size
    m = 18
    
    # new size
    n = 30
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(-0.01, 1.02, n+1)
    
    # some arbitrary distribution
    y_old = np.sin(x_old[:-1]*np.pi)
    
    # rebin
    y_new = rebin(x_old, y_old, x_new)
    
    # plot results ----------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure()
    edge_step(x_old, y_old, label='old')
    edge_step(x_new, y_new, label='new')
    
    plt.legend()
    plt.title("bin totals -- new is lower because its bins are narrower")
    
    plt.show()
