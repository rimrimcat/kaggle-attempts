from concurrent.futures.thread import ThreadPoolExecutor
from itertools import combinations
from typing import (
    Any,
    Optional,
    Sequence,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, hex2color, rgb2hex
from numpy.typing import NDArray
from scipy.stats import (
    kendalltau,
    kruskal,
    spearmanr,
    chi2_contingency,
    chisquare,
)
from sklearn.decomposition import PCA
from tqdm import tqdm

from __scripts__.typ import BasicDataType, StatResult

DEFAULT_FIGSIZE = (30, 16)
DEFAULT_FONTSIZE = 24  # TODO: IMPLEMENT
DEFAULT_COLOR_LIST = ["red", "orange", "blue", "green"]
DEFAULT_HEATMAP = LinearSegmentedColormap.from_list(
    "correlation", colors=["#4e0707", "#ffffff", "#000435"]
)

plt.rcParams.update({"font.size": DEFAULT_FONTSIZE})


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
        color_list = DEFAULT_COLOR_LIST

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


def find_corr(
    x: pd.Series,
    y: pd.Series,
    x_type: BasicDataType,
    y_type: BasicDataType,
):
    if x_type.is_categorical():
        x_cats = x.unique()
        x_cats.sort()

    if y_type.is_categorical():
        y_cats = y.unique()
        y_cats.sort()

    match x_type, y_type:
        # Main reference is doi:10.1177/8756479308317006
        case BasicDataType.CONTINUOUS, BasicDataType.CONTINUOUS:
            # spearmanr
            # can also use pearson r

            res_kw = spearmanr(x, y, alternative="two-sided")
            return StatResult(r"\rho", res_kw.statistic, res_kw.pvalue)

        case BasicDataType.ORDINAL, BasicDataType.ORDINAL:
            # kendall tau b or c
            # can also be spearmanr

            if len(x_cats) == len(y_cats):
                res_kw = kendalltau(x, y, variant="b", alternative="two-sided")
            else:
                res_kw = kendalltau(x, y, variant="c", alternative="two-sided")
            return StatResult(r"\tau", res_kw.statistic, res_kw.pvalue)

        case BasicDataType.NOMINAL, BasicDataType.NOMINAL:
            # goodman and kruskal lambda

            # table = pd.DataFrame(index=x_cats, columns=y_cats)
            # for x_cat in x_cats:
            #     for y_cat in y_cats:
            #         xdf = df.loc[df["x"] == x_cat]
            #         xydf = xdf.loc[xdf["y"] == y_cat]
            #         table.loc[x_cat, y_cat] = len(xydf)
            table = pd.crosstab(x, y)

            # sum of overall non-modal frequency
            y_sums = table.sum(axis=0)
            e1 = y_sums.loc[y_sums != y_sums.max()].sum()

            # sum of non-model frequency per independent variable
            x_maxes = table.max(axis=1).sum()
            e2 = table.sum().sum() - x_maxes
            gk_lambda = min((e1 - e2) / e1, 0)

            #! UNUSED, CRAMER'S
            res_chi2 = chi2_contingency(table)

            n = table.sum().sum()
            min_dim = min(table.shape) - 1
            cramers_v = np.sqrt(res_chi2.statistic / (n * min_dim))
            #! UNUSED

            if cramers_v > gk_lambda:
                return StatResult(r"\phi _c", cramers_v, res_chi2.pvalue)
            else:
                return StatResult(r"\lambda", gk_lambda, None)

        case BasicDataType.CONTINUOUS, BasicDataType.ORDINAL:
            # kendalltau b
            # can also be spearmanr

            res_kw = kendalltau(x, y, variant="b", alternative="two-sided")
            return StatResult(r"\tau", res_kw.statistic, res_kw.pvalue)

        case BasicDataType.NOMINAL, BasicDataType.ORDINAL:
            # chi-square and cramer's v
            contingency_table = pd.crosstab(x, y)
            res = chi2_contingency(contingency_table)

            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1

            if min_dim == 0:
                # No group
                return None

            cramers_v = np.sqrt(res.statistic / (n * min_dim))

            return StatResult(r"\phi _c", cramers_v, res.pvalue)

        case BasicDataType.CONTINUOUS, BasicDataType.NOMINAL:
            # point biserial
            # but probably will never encounter this?

            pass

    return None


def plot_counts(
    df: pd.DataFrame,
    num_plots_x: int,
    num_plots_y: int,
):
    dtype_dict = df.attrs["dtype_dict"]

    fig, axes = plt.subplots(num_plots_y, num_plots_x, figsize=DEFAULT_FIGSIZE)
    fig.tight_layout()
    axes = axes.flatten()

    def _make_subplot(_i, _x):
        if dtype_dict[_x].is_categorical():
            filter_df = df[_x].dropna().sort_values(ascending=True)

            sns.countplot(x=filter_df.astype(str), ax=axes[_i])
        else:
            filter_df = df[_x].dropna()
            # TODO: replace with distribution
            sns.violinplot(x=filter_df, ax=axes[_i])

        return _x

    # Create plots for each feature
    with ThreadPoolExecutor(max_workers=10) as executor:
        x_cols = df.columns.to_list()
        n = len(x_cols)
        tq = tqdm(
            executor.map(_make_subplot, range(n), x_cols),
            total=n,
            desc="Creating plots...",
        )

        for x in tq:
            tq.set_description(f"Creating count plot: {x}")

    n_diff = num_plots_x * num_plots_y - len(df.columns)
    if n_diff:
        for i in range(n_diff):
            fig.delaxes(axes[-(i + 1)])

    # plt.suptitle("Feature Violin Plots")
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=False)
    dtype_dict = df.attrs["dtype_dict"]


def tabulate_feature_corr(
    df: pd.DataFrame,
    alpha: float = 0.05,
):
    dtype_dict = df.attrs["dtype_dict"]

    cols = df.columns.to_list()
    n = len(cols)
    corr_df = pd.DataFrame(0.0, index=cols, columns=cols)

    pairs = combinations(cols, 2)

    for pair in pairs:
        x, y = pair

        filter_df = df[[x, y]].dropna()
        resxy = find_corr(
            filter_df[x],
            filter_df[y],
            dtype_dict[x].basic_type,
            dtype_dict[y].basic_type,
        )

        resyx = find_corr(
            filter_df[y],
            filter_df[x],
            dtype_dict[y].basic_type,
            dtype_dict[x].basic_type,
        )

        if resxy:
            corr_df.loc[x, y] = float(resxy.value)

        if resyx:
            corr_df.loc[y, x] = float(resyx.value)

    # annot_df = np.vectorize(make_annot)(corr_df)

    plt.figure(figsize=(8 + 2 * n, 4 + n))
    sns.heatmap(
        corr_df,
        fmt=".4f",
        annot=True,
        cmap=DEFAULT_HEATMAP,
        vmin=-1,
        vmax=1,
        mask=corr_df == 0,
    )
    plt.tick_params(
        axis="x",
        which="major",
        labelbottom=False,
        labeltop=True,
        rotation=70,
    )

    plt.show()


def plot_feature_label_corr(
    df: pd.DataFrame,
    x_cols: list[str],
    y: str,
    num_plots_x: int,
    num_plots_y: int,
    alpha: float = 0.05,
):
    dtype_dict = df.attrs["dtype_dict"]

    fig, axes = plt.subplots(num_plots_y, num_plots_x, figsize=DEFAULT_FIGSIZE)
    fig.tight_layout()
    axes = axes.flatten()

    # Made parallel
    def _make_subplot(_i: int, _x: str):
        filter_df = df[[_x, y]].dropna()
        filter_df.sort_values(by=y, ascending=False)

        x_dtype = dtype_dict[_x].basic_type
        y_dtype = dtype_dict[y].basic_type

        if x_dtype.is_categorical():
            x_cats: NDArray = filter_df[_x].unique()
            x_cats.sort()

            order = x_cats.tolist()
            plot_x = filter_df[_x].astype(str)
            # corrdf[x] = pd.factorize(filter_df[x])[0]
        else:
            order = None
            plot_x = filter_df[_x]

        if y_dtype.is_categorical():
            y_cats: NDArray = filter_df[y].unique()
            y_cats.sort()

            plot_y = filter_df[y].astype(str)
            # corrdf[y] = pd.factorize(filter_df[y])[0]
        else:
            plot_y = filter_df[y]

        sns.violinplot(
            x=plot_x,
            y=plot_y,
            order=order,
            ax=axes[_i],
        )

        res = find_corr(filter_df[_x], filter_df[y], x_dtype, y_dtype)
        if res:
            axes[_i].text(
                1.0,
                1.0,
                f"{res}",
                ha="right",
                va="top",
                transform=axes[_i].transAxes,
                bbox=dict(facecolor="white", alpha=0.5),
            )

        axes[_i].invert_yaxis()

        return _x

    # Create plots for each feature -> label
    with ThreadPoolExecutor(max_workers=10) as executor:
        n = len(x_cols)
        tq = tqdm(
            executor.map(_make_subplot, range(n), x_cols),
            total=n,
            desc="Creating plots...",
        )

        for x in tq:
            tq.set_description(f"Creating plot: {x} -> {y}")

    n_diff = num_plots_x * num_plots_y - len(x_cols)
    if n_diff:
        for i in range(n_diff):
            fig.delaxes(axes[-(i + 1)])

    # plt.suptitle("Bivariate Violin Plots")
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
