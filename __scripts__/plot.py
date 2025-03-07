from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from itertools import combinations
from math import ceil
from typing import (
    Any,
    Callable,
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
from pandas import Categorical
from scipy.signal import savgol_filter
from scipy.stats import (
    chi2_contingency,
    chisquare,
    kendalltau,
    kruskal,
    spearmanr,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from __scripts__.typ import BaseType, DataType, DataTypeDict, StatResult

DEFAULT_FIGSIZE = (32, 18)
DEFAULT_FONTSIZE = 24  # TODO: IMPLEMENT
DEFAULT_COLOR_LIST = ["red", "orange", "blue", "green"]
DEFAULT_HEATMAP = LinearSegmentedColormap.from_list(
    "correlation", colors=["#4e0707", "#ffffff", "#000435"]
)

plt.rcParams.update(
    {
        "font.size": DEFAULT_FONTSIZE,
        "axes.titlesize": DEFAULT_FONTSIZE,
        "axes.labelsize": DEFAULT_FONTSIZE,
    }
)
# sns.set_context("paper", rc={"font.size":DEFAULT_FONTSIZE,"axes.titlesize":8,"axes.labelsize":5})


class FigSize:
    DEFAULT = (20, 14)

    @staticmethod
    def from_subplots(x: int, y: int):
        return (12 * x, 5 * y)

    @staticmethod
    def from_table(rows: int, cols: int):
        return (8 + 2 * rows, 4 + cols)


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
    x_type: DataType,
    y_type: DataType,
):
    if x_type.categorical:
        x_cats = x.unique()
        if not isinstance(x_cats, Categorical):
            x_cats.sort()

    if y_type.categorical:
        y_cats = y.unique()
        if not isinstance(y_cats, Categorical):
            y_cats.sort()

    if x_type.continuous and y_type.continuous:
        # spearmanr
        # can also use pearson r

        res_kw = spearmanr(x, y, alternative="two-sided")
        return StatResult(r"\rho", res_kw.statistic, res_kw.pvalue)
    elif x_type.ordinal and y_type.ordinal:
        # kendall tau b or c
        # can also be spearmanr

        if len(x_cats) == len(y_cats):
            res_kw = kendalltau(x, y, variant="b", alternative="two-sided")
        else:
            res_kw = kendalltau(x, y, variant="c", alternative="two-sided")
        return StatResult(r"\tau", res_kw.statistic, res_kw.pvalue)
    elif x_type.nominal and y_type.nominal:
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

    elif x_type.continuous and y_type.ordinal:
        # kendalltau b
        # can also be spearmanr

        res_kw = kendalltau(x, y, variant="b", alternative="two-sided")
        return StatResult(r"\tau", res_kw.statistic, res_kw.pvalue)
    elif x_type.nominal and y_type.ordinal:
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
    elif x_type.continuous and y_type.ordinal:
        # point biserial
        # but probably will never encounter this?
        pass

    # match x_type, y_type:
    #     # Main reference is doi:10.1177/8756479308317006
    #     case BasicDataType.CONTINUOUS, BasicDataType.CONTINUOUS:
    #         # spearmanr
    #         # can also use pearson r

    #         res_kw = spearmanr(x, y, alternative="two-sided")
    #         return StatResult(r"\rho", res_kw.statistic, res_kw.pvalue)

    #     case BasicDataType.ORDINAL, BasicDataType.ORDINAL:
    #         # kendall tau b or c
    #         # can also be spearmanr

    #         if len(x_cats) == len(y_cats):
    #             res_kw = kendalltau(x, y, variant="b", alternative="two-sided")
    #         else:
    #             res_kw = kendalltau(x, y, variant="c", alternative="two-sided")
    #         return StatResult(r"\tau", res_kw.statistic, res_kw.pvalue)

    #     case BasicDataType.NOMINAL, BasicDataType.NOMINAL:
    #         # goodman and kruskal lambda

    #         # table = pd.DataFrame(index=x_cats, columns=y_cats)
    #         # for x_cat in x_cats:
    #         #     for y_cat in y_cats:
    #         #         xdf = df.loc[df["x"] == x_cat]
    #         #         xydf = xdf.loc[xdf["y"] == y_cat]
    #         #         table.loc[x_cat, y_cat] = len(xydf)
    #         table = pd.crosstab(x, y)

    #         # sum of overall non-modal frequency
    #         y_sums = table.sum(axis=0)
    #         e1 = y_sums.loc[y_sums != y_sums.max()].sum()

    #         # sum of non-model frequency per independent variable
    #         x_maxes = table.max(axis=1).sum()
    #         e2 = table.sum().sum() - x_maxes
    #         gk_lambda = min((e1 - e2) / e1, 0)

    #         #! UNUSED, CRAMER'S
    #         res_chi2 = chi2_contingency(table)

    #         n = table.sum().sum()
    #         min_dim = min(table.shape) - 1
    #         cramers_v = np.sqrt(res_chi2.statistic / (n * min_dim))
    #         #! UNUSED

    #         if cramers_v > gk_lambda:
    #             return StatResult(r"\phi _c", cramers_v, res_chi2.pvalue)
    #         else:
    #             return StatResult(r"\lambda", gk_lambda, None)

    #     case BasicDataType.CONTINUOUS, BasicDataType.ORDINAL:
    #         # kendalltau b
    #         # can also be spearmanr

    #         res_kw = kendalltau(x, y, variant="b", alternative="two-sided")
    #         return StatResult(r"\tau", res_kw.statistic, res_kw.pvalue)

    #     case BasicDataType.NOMINAL, BasicDataType.ORDINAL:
    #         # chi-square and cramer's v
    #         contingency_table = pd.crosstab(x, y)
    #         res = chi2_contingency(contingency_table)

    #         n = contingency_table.sum().sum()
    #         min_dim = min(contingency_table.shape) - 1

    #         if min_dim == 0:
    #             # No group
    #             return None

    #         cramers_v = np.sqrt(res.statistic / (n * min_dim))

    #         return StatResult(r"\phi _c", cramers_v, res.pvalue)

    #     case BasicDataType.CONTINUOUS, BasicDataType.NOMINAL:
    #         # point biserial
    #         # but probably will never encounter this?

    #         pass

    return None


def plot_counts(
    df: pd.DataFrame,
    num_plots_x: int = 4,
    num_plots_y: Optional[int] = None,
):
    dtype_dict = df.attrs["dtype_dict"]

    n_plots = len(df.columns)

    if num_plots_y is None:
        num_plots_y = ceil((n_plots + 1) / num_plots_x)

    fig, axes = plt.subplots(num_plots_y, num_plots_x, figsize=DEFAULT_FIGSIZE)
    fig.tight_layout()
    axes = axes.flatten()

    def _make_subplot(_i, _x):
        if dtype_dict[_x].categorical:
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

    n_diff = num_plots_x * num_plots_y - n_plots
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
            dtype_dict[x],
            dtype_dict[y],
        )

        resyx = find_corr(
            filter_df[y],
            filter_df[x],
            dtype_dict[y],
            dtype_dict[x],
        )

        if resxy:
            corr_df.loc[x, y] = float(resxy.value)

        if resyx:
            corr_df.loc[y, x] = float(resyx.value)

    # annot_df = np.vectorize(make_annot)(corr_df)

    plt.figure(figsize=FigSize.from_table(n, n))
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

    plt.show(block=False)


def plot_feature_label_corr(
    df: pd.DataFrame,
    y: str,
    x_cols: Optional[list[str]] = None,
    num_plots_x: int = 4,
    num_plots_y: Optional[int] = None,
    alpha: float = 0.05,
):
    dtype_dict: DataTypeDict = df.attrs["dtype_dict"]

    if not x_cols:
        x_cols_ = df.columns.drop(y).to_list()
        x_cols = [col for col in x_cols_ if not dtype_dict[col].base == BaseType.UNK]

    if not num_plots_y:
        num_plots_y = ceil(len(x_cols) / num_plots_x)

    fig, axes = plt.subplots(
        num_plots_y,
        num_plots_x,
        # figsize=DEFAULT_FIGSIZE,
        figsize=FigSize.from_subplots(num_plots_x, num_plots_y),
    )
    fig.tight_layout()
    axes = axes.flatten()

    # Made parallel
    def _make_subplot(_i: int, _x: str):
        filter_df = df[[_x, y]].dropna()
        filter_df.sort_values(by=[y, _x], ascending=False, inplace=True)

        x_dtype = dtype_dict[_x]
        y_dtype = dtype_dict[y]

        if x_dtype.categorical:
            x_cats: NDArray = filter_df[_x].unique()
            if not isinstance(x_cats, Categorical):
                x_cats.sort()

            order = x_cats.tolist()
            plot_x = filter_df[_x].astype(str)
        else:
            order = None
            plot_x = filter_df[_x]

        if y_dtype.categorical:
            y_cats: NDArray = filter_df[y].unique()
            if not isinstance(y_cats, Categorical):
                y_cats.sort()

            plot_y = filter_df[y].astype(str)
        else:
            plot_y = filter_df[y]

        if x_dtype.continuous and y_dtype.categorical:
            sns.violinplot(
                x=plot_x,
                y=plot_y,
                order=order,
                ax=axes[_i],
            )
        elif x_dtype.categorical and y_dtype.categorical:
            table = pd.crosstab(
                index=filter_df[y],
                columns=filter_df[_x],
            )
            table.sort_index(axis="index", ascending=False, inplace=True)

            tab = axes[_i].table(
                cellText=table.astype(str).to_numpy().tolist()
                + [[f"{__txt}" for __txt in table.columns.tolist()]],
                cellLoc="center",
                rowLoc="right",
                bbox=[0.1, 0, 0.9, 0.8],
                rowLabels=[f"{__txt}  " for __txt in table.index.tolist()] + [""],
            )
            tab.auto_set_font_size(False)
            tab.set_fontsize(DEFAULT_FONTSIZE)

            # plt.tick_params(
            #     axis="x",
            #     which="major",
            #     labelbottom=False,
            #     labeltop=True,
            #     rotation=70,
            # )
            # axes[_i].tick_params(
            #
            #     rotation=70,
            # )

            axes[_i].set_xlabel(_x)
            axes[_i].set_ylabel(y)

            axes[_i].set_xticks([])
            axes[_i].set_yticks([])

            axes[_i].spines["top"].set_visible(False)
            axes[_i].spines["right"].set_visible(False)
            axes[_i].spines["left"].set_visible(False)
            axes[_i].spines["bottom"].set_visible(False)

            for key, cell in tab.get_celld().items():
                cell.set_linewidth(0)

            for i__ in range(len(y_cats)):
                for j__ in range(len(x_cats)):
                    data_cell = tab[(i__, j__)]
                    data_cell.set_linewidth(1.5)

            cat_lengths = []
            for j__ in range(len(x_cats)):
                data_cell = tab[(len(y_cats), j__)]
                data_text = data_cell.get_text().get_text()
                cat_lengths.append(len(data_text))

            if max(cat_lengths) > DEFAULT_FONTSIZE / len(x_cats):
                new_max_len = DEFAULT_FONTSIZE / len(x_cats)
                new_font_size = new_max_len * len(x_cats) / 2
                for j__ in range(len(x_cats)):
                    data_cell = tab[(len(y_cats), j__)]
                    data_cell.set_text_props(
                        rotation=-60,
                        fontsize=new_font_size,
                    )

        else:
            raise NotImplementedError(
                f"Ploting feature with type {x_dtype._name_} and label with type {y_dtype._name_} not implemented yet!"
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


def plot_ts_clf_label(
    df: pd.DataFrame,
    label: str,
    ts: Optional[str] = None,
    features: Optional[list[str]] = None,
    test_df: Optional[pd.DataFrame] = None,
    smoothing: Optional[Callable] = partial(savgol_filter, polyorder=2),
    window_frac: float = 0.05,
    smooth_every: float = 0.025,
):

    if ts is None:
        # Get the first TimeData dtype
        dtype_dict = df.attrs["dtype_dict"]

        for col in df.columns:
            if dtype_dict[col] == DataType.TIME_DATA:
                ts = col
                break
        else:
            raise ValueError("Provided dataframe doesn't contain time series column!")

    if features is None:
        features = []
        # Get the first three features
        for col in df.columns:
            if col not in [ts, label]:
                if len(features) < 3:
                    features.append(col)
                else:
                    break
    feature_markers = ["o", "^", "+"]

    df = df.sort_values(by=ts, ascending=True)

    X = df[ts]

    # Decimal X, for plotting
    X_dec = (X - X.iloc[0]) / (X.iloc[-1] - X.iloc[0])

    Y = df[label]

    Y_cols = assign_colors(Y.unique())

    fig = plt.figure(figsize=FigSize.DEFAULT)

    window_length = int(window_frac * len(X))
    plot_length = int(smooth_every * len(X))

    p_x = X_dec.to_numpy().ravel()

    for i, f in enumerate(features):
        scaler = MinMaxScaler()
        f_scaled = scaler.fit_transform(df[f].to_numpy().reshape(-1, 1))

        p_f = f_scaled.ravel()
        if smoothing:
            plt.plot(
                p_x[::plot_length],
                smoothing(p_f, window_length)[::plot_length],
                marker=feature_markers[i],
                label=f,
            )

            plt.scatter(
                p_x,
                p_f,
                marker=".",
                alpha=0.2,
                label=None,
            )

        else:
            plt.plot(
                p_x,
                p_f,
                marker=feature_markers[i],
            )

    plt.plot(
        p_x,
        Y.to_numpy().ravel() * 0.1,
        label=None,
    )

    # Add classifications
    plt.legend()
    plt.show()


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

    fig = plt.figure(figsize=DEFAULT_FIGSIZE)

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
        fig = plt.figure(figsize=DEFAULT_FIGSIZE)
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

    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
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


def plot_ts(X, y):
    # TODO: plot time series for visualization
    pass
