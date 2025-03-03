from itertools import combinations
from math import comb
from typing import Optional, Union

import numpy as np
from sklearn.model_selection._split import _BaseKFold


def get_ranges(X: np.ndarray):
    X = X.astype(int)
    disc_idx = np.where(np.diff(X) != 1)[0].astype(int)

    r_list = []
    if len(disc_idx) > 0:
        for i, idx in enumerate(disc_idx):
            if i == 0:
                r_list.append((X[0].item(), X[idx].item()))
                last_ = idx + 1
            else:
                r_list.append((X[last_].item(), X[idx].item()))
                last_ = idx + 1

        r_list.append((X[last_].item(), X[-1].item()))
    elif len(X) > 1:
        r_list.append((X[0].item(), X[-1].item()))
    else:
        r_list = []
    return r_list


class PurgedKFold(_BaseKFold):
    def __init__(
        self,
        n_splits=5,
        delta: Union[int, np.ndarray] = 5,
        frac_embargo: float = 0.0,
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)

        self.delta = delta
        self.frac_embargo = frac_embargo

    def split(
        self,
        X,
        y=None,
        groups=None,
    ):
        n_samples = X.shape[0]

        if isinstance(self.delta, int):
            delta = (np.ones(n_samples) * self.delta).astype(int)

        indices = np.arange(n_samples)
        len_embargo = int(n_samples * self.frac_embargo)

        fold_blocks = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(n_samples), self.n_splits)
        ]

        # print("fold_blocks:")
        print(fold_blocks)

        for i, j in fold_blocks:
            test_indices = indices[i:j]
            right_idx_start = test_indices[-1] + delta[test_indices[-1]]
            # print("test_indices:", get_ranges(test_indices), "j", j)

            ## Train indices before test_indices
            if i != 0:
                train_indices = indices[: i - delta[i] + 1]
            else:
                train_indices = np.array([])
            # print("train_indices left:", get_ranges(train_indices))

            if right_idx_start < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate(
                    (train_indices, indices[right_idx_start + len_embargo :])
                )
                # print(
                #     "train_indices_right:",
                #     get_ranges(indices[right_idx_start + len_embargo :]),
                # )

            # print("")
            # print("Train:", get_ranges(train_indices))
            # print("Test:", get_ranges(test_indices))
            yield train_indices, test_indices


class CombinatorialPurgedKFold(_BaseKFold):
    def __init__(
        self,
        n_splits: Optional[int] = None,
        n_groups: Optional[int] = None,
        test_groups: Optional[int] = None,
        delta: Union[int, np.ndarray] = 5,
        frac_embargo: float = 0.0,
    ):

        if not n_splits:
            if not n_groups:
                n_groups = 4
                # int(ts.shape[0] / 150)
            if not test_groups:
                test_groups = 2

            self.n_groups = n_groups
            self.test_groups = test_groups

            n_splits = comb(n_groups, test_groups)

        else:
            n_groups = n_splits
            test_groups = 1

        super().__init__(n_splits, shuffle=False, random_state=None)
        self.delta = delta
        self.frac_embargo = frac_embargo
        self.n_groups = n_groups
        self.test_groups = test_groups

    def split(
        self,
        X,
        y=None,
        groups=None,
    ):
        n_samples = X.shape[0]

        if isinstance(self.delta, int):
            delta = (np.ones(n_samples) * self.delta).astype(int)

        indices = np.arange(n_samples)
        len_embargo = int(n_samples * self.frac_embargo)

        fold_blocks = np.array(
            [
                (i[0], i[-1] + 1)
                for i in np.array_split(np.arange(n_samples), self.n_splits)
            ]
        )
        # n+1 blocks
        combs = list(combinations(range(self.n_groups), self.test_groups))
        # print("Fold_blocks:", fold_blocks)

        for comb_tup in combs:
            # print("comb_tup:", comb_tup)
            test_indices = np.array([])
            train_indices = np.array([])

            for test_grp in comb_tup:
                test_indices = np.append(
                    test_indices,
                    np.arange(fold_blocks[test_grp][0], fold_blocks[test_grp][1]),
                )
            test_ranges = get_ranges(test_indices)

            for i_, test_range in enumerate(test_ranges):
                # print("Iteration:", i_)
                if i_ == 0:
                    # Skip if test set is first block
                    if test_range[0] == 0:
                        # print("No left train set")
                        pass
                    else:
                        # print("Left train set")

                        train_idx_end = test_range[0] - delta[test_range[0]] + 1
                        train_indices = np.concatenate(
                            (
                                train_indices,
                                indices[:train_idx_end],
                            )
                        )

                if i_ == len(test_ranges) - 1:
                    if test_range[1] >= X.shape[0]:
                        # No right train (with embargo)
                        # print("No right train with embargo")
                        pass
                    else:
                        # print("Right train with embargo")

                        train_idx_start = test_range[1] + delta[test_range[1]]
                        train_indices = np.concatenate(
                            (train_indices, indices[train_idx_start + len_embargo :])
                        )

                else:
                    # Inbetween
                    # print("Do inbetweens")

                    train_idx_start = test_range[1] + delta[test_range[1]]
                    train_idx_end = (
                        test_ranges[i_ + 1][0] - delta[test_ranges[i_ + 1][0]] + 1
                    )

                    train_indices = np.concatenate(
                        (
                            train_indices,
                            indices[train_idx_start + len_embargo : train_idx_end],
                        )
                    )

            # print("Test", get_ranges(test_indices))
            # print("Train:", get_ranges(train_indices))
            yield train_indices, test_indices
