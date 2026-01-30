#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 21:10:50 2026

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

def plot_ifgs_corrected_residual(
    A,
    B,
    plot_n=-1,
    cmap="viridis",
    titles=("A", "B", "Residual (A - B)"),
    robust=False,
    robust_pct=(2, 98),
    show_axes=False,
):
    """
    Plot A[plot_n], B[plot_n], and residual A[plot_n] - B[plot_n].

    Parameters
    ----------
    A, B : array-like, shape (n, ny, nx)
        Input 3D tensors.
    plot_n : int
        Slice index along axis 0 to plot.
    cmap : str
        Matplotlib colormap name.
    titles : tuple[str, str, str]
        Titles for the three panels.
    robust : bool
        If True, compute vmin/vmax using percentiles (helps with outliers).
    robust_pct : tuple[float, float]
        Percentiles for robust scaling, e.g. (2, 98).
    show_axes : bool
        If False, hide axis ticks/frames for cleaner image panels.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """

    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("A and B must be 3D arrays with shape (n, ny, nx).")
    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape. Got {A.shape} vs {B.shape}.")

    a2 = A[plot_n]
    b2 = B[plot_n]
    r2 = a2 - b2

    # Shared scaling for first two panels
    if robust:
        lo, hi = robust_pct
        vmin_ab, vmax_ab = np.nanpercentile(np.concatenate([a2.ravel(), b2.ravel()]), [lo, hi])
        vmin_r,  vmax_r  = np.nanpercentile(r2.ravel(), [lo, hi])
    else:
        vmin_ab = np.nanmin([np.nanmin(a2), np.nanmin(b2)])
        vmax_ab = np.nanmax([np.nanmax(a2), np.nanmax(b2)])
        vmin_r  = np.nanmin(r2)
        vmax_r  = np.nanmax(r2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(a2, cmap=cmap, vmin=vmin_ab, vmax=vmax_ab, origin="upper")
    im1 = axes[1].imshow(b2, cmap=cmap, vmin=vmin_ab, vmax=vmax_ab, origin="upper")
    im2 = axes[2].imshow(r2, cmap=cmap, vmin=vmin_r,  vmax=vmax_r,  origin="upper")

    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])
    axes[2].set_title(titles[2])

    if not show_axes:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

    # One horizontal colorbar shared under the first two subplots
    cbar_ab = fig.colorbar(
        im0,
        ax=[axes[0], axes[1]],
        orientation="horizontal",
        fraction=0.06,
        pad=0.08,
        aspect=40,
    )
    cbar_ab.set_label("Original and Corrected")

    # Separate colorbar for residual (own vmin/vmax)
    cbar_r = fig.colorbar(
        im2,
        ax=axes[2],
        orientation="horizontal",
        fraction=0.06,
        pad=0.08,
        aspect=40,
    )
    cbar_r.set_label("Residual ")

    return fig, axes
