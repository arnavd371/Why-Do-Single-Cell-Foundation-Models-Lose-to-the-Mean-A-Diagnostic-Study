"""Train / validation / test splits for the Replogle perturbation dataset.

CRITICAL DESIGN NOTE
--------------------
Splits are performed at the *perturbation* level, **not** at the cell level.
Splitting by cell would leak information — the model would see cells from the
same perturbation during training and evaluation, artificially inflating scores.
By splitting perturbation names we ensure that test perturbations are
completely unseen during training.
"""

import random
from typing import List, Tuple

import anndata


def train_val_test_split(
    perturbations: List[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split perturbation names into train / validation / test sets.

    The split is performed on perturbation *names* (not on individual cells).
    This prevents data leakage: a model trained with this split will never
    have seen any cell from a test perturbation at training time.

    Parameters
    ----------
    perturbations:
        Sorted list of unique perturbation names (from
        ``list_perturbations``).
    train_frac:
        Fraction of perturbations for the training set.
    val_frac:
        Fraction for the validation set.  The remainder goes to the test set.
    seed:
        Random seed for reproducible shuffling.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        ``(train_perts, val_perts, test_perts)`` — non-overlapping lists of
        perturbation names.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0.")

    perts = list(perturbations)
    rng = random.Random(seed)
    rng.shuffle(perts)

    n = len(perts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_perts = perts[:n_train]
    val_perts = perts[n_train : n_train + n_val]
    test_perts = perts[n_train + n_val :]

    return train_perts, val_perts, test_perts


def get_split_data(
    adata: anndata.AnnData,
    split: str,
    train_perts: List[str],
    val_perts: List[str],
    test_perts: List[str],
) -> anndata.AnnData:
    """Return cells that belong to the perturbations in a given split.

    Parameters
    ----------
    adata:
        Full preprocessed AnnData object.
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    train_perts, val_perts, test_perts:
        Perturbation name lists from :func:`train_val_test_split`.

    Returns
    -------
    anndata.AnnData
        Subset of *adata* containing only cells whose perturbation belongs
        to the requested split.
    """
    split_map = {
        "train": train_perts,
        "val": val_perts,
        "test": test_perts,
    }
    if split not in split_map:
        raise ValueError(f"split must be one of {list(split_map.keys())}, got {split!r}.")

    perts_in_split = set(split_map[split])
    mask = adata.obs["perturbation"].isin(perts_in_split)
    return adata[mask].copy()
