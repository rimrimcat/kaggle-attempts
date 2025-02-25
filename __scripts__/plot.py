from typing import (
    Any,
    Optional,
    Sequence,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, hex2color, rgb2hex
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from scipy.stats import chisquare, chisquare

from __scripts__.types import BasicDataType, DataTypeDict


def assign_colors(class_list: Sequence[Any], color_list=None):
    """
    Assign colors to classes either directly from color_list or through interpolation.

    Parameters:
    -----------
    class_list : list or array-like
        List of unique class values
    color_list : list, optional
        List of colors in hex or name format. Default is ["red","orange","blue","green"]

    Returns:
    --------
    list
        List of colors in hex format corresponding to each class
    """
    if color_list is None:
        color_list = ["red", "orange", "blue", "green"]

    n_classes = len(class_list)
    n_colors = len(color_list)

    # Convert all colors to RGB format for consistent handling
    rgb_colors = [hex2color(rgb2hex(color)) for color in color_list]

    # Case 1: If number of classes <= number of colors, assign directly
    if n_classes <= n_colors:
        return [rgb2hex(rgb_colors[i]) for i in range(n_classes)]

    # Case 2: If number of classes > number of colors, interpolate
    # Create evenly spaced values between 0 and 1
    values = np.linspace(0, 1, n_classes)

    # Create a colormap from the provided colors
    cmap = LinearSegmentedColormap.from_list("custom", rgb_colors)

    # Get interpolated colors
    interpolated_colors = [rgb2hex(cmap(v)) for v in values]

    return interpolated_colors


def plot_monovariate_violin(
    df: pd.DataFrame,
    dtype_dict: DataTypeDict,
    num_plots_x: int,
    num_plots_y: int,
):
    fig, axes = plt.subplots(num_plots_y, num_plots_x, figsize=(12, 8))
    fig.tight_layout()
    axes = axes.flatten()

    for i, x in enumerate(df.columns.to_list()):
        if dtype_dict[x].is_categorical():
            filter_df = df[x].dropna().sort_values(ascending=True)

            sns.violinplot(x=filter_df.astype(str), ax=axes[i])
        else:
            filter_df = df[x].dropna()
            sns.violinplot(x=filter_df, ax=axes[i])

    n_diff = num_plots_x * num_plots_y - len(df.columns)
    if n_diff:
        for i in range(n_diff):
            fig.delaxes(axes[-(i + 1)])

    plt.suptitle("Feature Violin Plots")
    plt.show(block=False)


def plot_counts(
    df: pd.DataFrame,
    dtype_dict: DataTypeDict,
    num_plots_x: int,
    num_plots_y: int,
):
    fig, axes = plt.subplots(num_plots_y, num_plots_x, figsize=(12, 8))
    fig.tight_layout()
    axes = axes.flatten()

    for i, x in enumerate(df.columns.to_list()):
        if dtype_dict[x].is_categorical():
            filter_df = df[x].dropna().sort_values(ascending=True)

            sns.countplot(x=filter_df.astype(str), ax=axes[i])
        else:
            filter_df = df[x].dropna()
            # TODO: replace with distribution
            sns.violinplot(x=filter_df, ax=axes[i])

    n_diff = num_plots_x * num_plots_y - len(df.columns)
    if n_diff:
        for i in range(n_diff):
            fig.delaxes(axes[-(i + 1)])

    plt.suptitle("Feature Violin Plots")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=False)


def plot_bivariate_violin(
    df: pd.DataFrame,
    dtype_dict: DataTypeDict,
    x_cols: list[str],
    y: str,
    num_plots_x: int,
    num_plots_y: int,
):
    fig, axes = plt.subplots(num_plots_y, num_plots_x, figsize=(12, 8))
    fig.tight_layout()
    axes = axes.flatten()

    # Calculate expected value
    if dtype_dict[y].is_categorical():
        valc = df[[y]].value_counts()
        freq_exp_frac = (valc / valc.sum()).to_numpy()
    else:
        freq_exp_frac = None

    # Create plots for each feature -> label
    for i, x in enumerate(x_cols):
        filter_df = df[[x, y]].dropna()
        filter_df.sort_values(by=y, ascending=False)
        corr = None

        x_dtype = dtype_dict[x].basic_type
        y_dtype = dtype_dict[y].basic_type

        if x_dtype.is_ordinal() and y_dtype.is_ordinal():
            # Perform simple linear correlation
            corrdf = filter_df.copy()

            if x_dtype.is_categorical():
                x_cats: NDArray = filter_df[x].unique()
                x_cats.sort()

                order = x_cats.tolist()
                plot_x = filter_df[x].astype(str)
                corrdf[x] = pd.factorize(filter_df[x])[0]
            else:
                order = None
                plot_x = filter_df[x]

            if y_dtype.is_categorical():
                y_cats: NDArray = filter_df[y].unique()
                y_cats.sort()

                plot_y = filter_df[y].astype(str)
                corrdf[y] = pd.factorize(filter_df[y])[0]
            else:
                plot_y = filter_df[y]

            sns.violinplot(
                x=plot_x,
                y=plot_y,
                order=order,
                ax=axes[i],
            )

            # Print linear correlation
            corr = corrdf.corr()[x][y]
            axes[i].text(
                1.0,
                1.0,
                f"CORR: {round(corr, 4)}",
                ha="right",
                va="top",
                transform=axes[i].transAxes,
                bbox=dict(facecolor="white", alpha=0.5),
            )
        else:
            # Chi square

            if x_dtype.is_categorical() and y_dtype.is_categorical():
                x_cats: NDArray = filter_df[x].unique()
                x_cats.sort()

                p_vals = []
                for cat in x_cats.tolist():
                    print("Current cat:", cat)
                    freq_obs = filter_df.loc[filter_df[x] == cat, y].value_counts()
                    freq_exp = freq_exp_frac * freq_obs.sum()
                    p_val = chisquare(
                        f_obs=freq_obs,
                        f_exp=freq_exp,
                    )
                    print("got p_val", p_val)
                    p_vals.append(p_val)

                    # TODO: SKIP IF NOT ENOUGH OBSERVATIONS

            else:
                # this is default behavior, replace it
                sns.violinplot(
                    x=filter_df[x],
                    y=filter_df[y].astype(str),
                    ax=axes[i],
                )

        axes[i].invert_yaxis()

    n_diff = num_plots_x * num_plots_y - len(x_cols)
    if n_diff:
        for i in range(n_diff):
            fig.delaxes(axes[-(i + 1)])

    plt.suptitle("Bivariate Violin Plots")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=False)


def plot_scree(
    pca: PCA,
    component_display_limit: int = 7,
):
    """
    Creates a scree plot

    Args:
        pca: Fitted PCA.
        component_display_limit: Maximum number of components to display on scree plot.

    Returns:
        None
    """
    n_components = pca.n_components_
    explained_variance_ratio = pca.explained_variance_ratio_

    if n_components > component_display_limit:
        # Show main 7, others can be combined
        l_1 = component_display_limit - 1

        x = [str(i) for i in range(1, component_display_limit)] + [f"...{n_components}"]
        y = explained_variance_ratio[:l_1].tolist() + [
            explained_variance_ratio[l_1:].sum()
        ]

    else:
        x = [str(i) for i in range(1, n_components + 1)]
        y = explained_variance_ratio

    # Bar Plot
    sns.barplot(
        x=x,
        y=y,
        width=1.5,
        gap=0.5,
    )

    # Cumulative
    cumy = list(np.cumsum(y))
    plt.plot(
        cumy,
        marker="o",
        linestyle="-",
        color="gray",
    )

    for i, height in enumerate(y):
        _x = i + 1 - 2.0 / 2

        if i == 0:
            text_height = height + 0.05
        else:
            text_height = height + 0.01

            plt.text(
                _x,
                cumy[i] + 0.02,
                f"{cumy[i]:.2f}",
                ha="center",
                va="bottom",
                color="gray",
            )

        plt.text(
            _x,
            text_height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.ylim([0, 1.10])
    plt.xlabel("Principal Component")
    plt.show(block=False)


def plot_principal_components(
    pc1: NDArray,
    pc2: NDArray,
    pc3: Optional[NDArray] = None,
    classes: Optional[NDArray] = None,
    class_names: Optional[list[str]] = None,
    class_colors: Optional[list[Any]] = ["red", "orange", "blue", "green"],
):
    """
    Plots the PCA projection

    Args:
        pc1: Values for first principal component
        pc2: Values for second principal component
        pc3: Values for third principal component
        classes: Specifies the class of the data point


    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.
    """

    if not pc3:
        # 2D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        if classes is not None and classes.shape[1] == 1:
            classes = classes.ravel()
            uniq_classes = list(set(classes.tolist()))
            colors = assign_colors(uniq_classes)

            for i, _cls in enumerate(uniq_classes):
                inds = np.where(classes == _cls)[0]
                ax.scatter(pc1[inds], pc2[inds], c=[colors[i]], alpha=0.7)

            if class_names is not None:
                ax.legend(class_names)
            else:
                ax.legend([f"Class {x + 1}" for x in range(len(uniq_classes))])

        else:
            ax.scatter(pc1, pc2, alpha=0.7)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("2D PCA Projection")
        ax.grid(True)
        plt.show(block=False)

    else:
        raise NotImplementedError()


def plot_pca_loadings(
    pca: PCA,
    feature_names: Optional[list[str]] = None,
    include_above: float = 0.25,
):
    components: NDArray = pca.components_

    if not feature_names:
        feature_names = [str(i + 1) for i in range(len(components))]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for i in range(components.shape[0]):
        pc1 = components[i, 0]
        pc2 = components[i, 1]

        # if pc1 < include_above and pc2 < include_above:
        if np.sqrt(pc1**2 + pc2**2) < include_above:
            continue

        ax.arrow(
            0,
            0,
            pc1,
            pc2,
            color="r",
            alpha=0.5,
            head_width=0.05,
            head_length=0.1,
        )
        ax.text(
            pc1 * 1.15,
            pc2 * 1.15,
            feature_names[i],
            color="g",
            ha="center",
            va="center",
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Loadings Plot")
    ax.grid(True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    plt.show(block=False)
