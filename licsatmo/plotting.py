#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 21:10:50 2026

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime


def plot_ifgs_corrected_residual(
    A,
    B,
    plot_n=-1,
    cmap="viridis",
    titles=("A", "B", "Residual (A - B)"),
    robust=False,
    robust_pct=(2, 98),
    show_axes=False,
    outfile=None,
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
    outfile | None or Path
        Filename to save png to.  

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
    
    # if a filename provided, save as that 
    if outfile is not None:
        fig.savefig(outfile)

    return fig, axes


def plot_pixel_history_two_r3(
    r3_a,
    r3_b,
    x,
    y,
    *,
    acq_dates=None,                 # <- NEW: list/array of 'YYYYMMDD' (or datetime-like)
    labels=("Signal A", "Signal B"),
    window=9,
    center=True,
    show_scatter=True,
    scatter_kwargs=None,
    line_kwargs=None,
    date_fmt="%Y-%m-%d",
    rotate_dates=30,
    outfile=None,
):
    """
    Plot the time history at one pixel ([:, y, x]) for two 3D arrays.

    Uses acquisition dates on the x-axis if provided (acq_dates).

    Layout:
      - Row 1: two axes side-by-side:
          each shows scatter points + moving-average smoothed line
      - Row 2: one axis spanning both columns, twice the height:
          shows only the two smoothed lines
    """
    a = np.asarray(r3_a)
    b = np.asarray(r3_b)

    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Both inputs must be 3D arrays with shape (t, ny, nx).")
    if a.shape != b.shape:
        raise ValueError(f"Inputs must have the same shape. Got {a.shape} vs {b.shape}.")

    t, ny, nx = a.shape
    if not (0 <= x < nx and 0 <= y < ny):
        raise IndexError(f"(x, y)=({x},{y}) out of bounds for shape (t, ny, nx)={a.shape}.")

    # ---- X axis: time index or acquisition dates ----
    if acq_dates is None:
        xx = np.arange(t)
        use_dates = False
    else:
        if len(acq_dates) != t:
            raise ValueError(f"acq_dates length must equal t={t}. Got {len(acq_dates)}.")
        
        xx = [datetime.strptime(d, "%Y%m%d") for d in acq_dates]
        use_dates = True

    # Extract time series
    ts_a = a[:, y, x].astype(float)
    ts_b = b[:, y, x].astype(float)

    # Moving average that respects NaNs (simple + robust enough)
    def moving_average_nan(x, w, centered=True):
        x = np.asarray(x, dtype=float)
        if w <= 1:
            return x.copy()
        if w % 2 == 0:
            w += 1  # make odd for nicer centering

        valid = np.isfinite(x).astype(float)
        x0 = np.where(np.isfinite(x), x, 0.0)

        kernel = np.ones(w, dtype=float)
        num = np.convolve(x0, kernel, mode="same")
        den = np.convolve(valid, kernel, mode="same")
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

        if centered:
            return out

        # trailing version: shift so each point uses previous w samples
        shift = (w - 1) // 2
        return np.roll(out, shift)

    sm_a = moving_average_nan(ts_a, window, centered=center)
    sm_b = moving_average_nan(ts_b, window, centered=center)

    scatter_kwargs = {} if scatter_kwargs is None else dict(scatter_kwargs)
    line_kwargs = {} if line_kwargs is None else dict(line_kwargs)

    # ---- Figure layout (bottom is twice as high) ----
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # ---- Top row: each signal scatter + smoothed line ----
    if show_scatter:
        ax1.scatter(xx, ts_a, s=10, alpha=0.6, **scatter_kwargs)
    ax1.plot(xx, sm_a, linewidth=2, **line_kwargs)
    ax1.set_title(f"{labels[0]} at (x={x}, y={y})")
    ax1.set_xlabel("date" if use_dates else "time index")
    ax1.set_ylabel("value")
    ax1.grid(True, alpha=0.3)

    if show_scatter:
        ax2.scatter(xx, ts_b, s=10, alpha=0.6, **scatter_kwargs)
    ax2.plot(xx, sm_b, linewidth=2, **line_kwargs)
    ax2.set_title(f"{labels[1]} at (x={x}, y={y})")
    ax2.set_xlabel("date" if use_dates else "time index")
    ax2.set_ylabel("Displacement (m)")
    ax2.grid(True, alpha=0.3)

    # ---- Bottom row: only smoothed lines (both together) ----
    ax3.plot(xx, sm_a, linewidth=2, label=f"{labels[0]} (smoothed)", **line_kwargs)
    ax3.plot(xx, sm_b, linewidth=2, label=f"{labels[1]} (smoothed)", **line_kwargs)
    ax3.set_title(f"Smoothed comparison at (x={x}, y={y})")
    ax3.set_xlabel("date" if use_dates else "time index")
    ax3.set_ylabel("Displacement (m)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # ---- If using dates, format ticks nicely ----
    if use_dates:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)

        # for ax in (ax1, ax2, ax3):
        #     ax.xaxis.set_major_locator(locator)
        #     ax.xaxis.set_major_formatter(formatter)
        #     for label in ax.get_xticklabels():
        #         label.set_rotation(rotate_dates)
        #         label.set_ha("right")

    axes = {"top_a": ax1, "top_b": ax2, "bottom": ax3}
    
    # if a filename provided, save as that 
    if outfile is not None:
        fig.savefig(outfile)
    
    return fig, axes
