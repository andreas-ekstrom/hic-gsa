import numpy as np
from typing import List, Tuple, Optional

def read_list_of_points(
    filename: str,
    order_parameters: List[str],
    skip_first: bool = True,
    return_names: bool = False
) -> np.ndarray | Tuple[np.ndarray, List[str]]:
    """
    Read a text file with a header line listing LEC names (prefixed by '#'),
    then return the data matrix with columns reordered to match `order_parameters`.
    
    Any names in `order_parameters` that are NOT present in the file header are
    skipped, and a warning is printed (e.g., 'const' not present -> skipped).

    Parameters
    ----------
    filename : str
        Path to the file.
    order_parameters : list of str
        Desired column order. Names not found in the file are skipped (with warning).
    skip_first : bool, default True
        Backwards-compat flag; ignored if a header line beginning with '#' is present.
        If no header is found and this is True, the first line will be skipped.
    return_names : bool, default False
        If True, also return the list of column names that were actually used (in order).

    Returns
    -------
    data : (n_rows, n_cols_kept) ndarray
        Data with columns reordered to match the subset of `order_parameters` that
        were found in the file header.
    names_used : list of str
        Only returned if `return_names=True`. The column names corresponding to
        the returned data (same order as columns in `data`).
    """
    # --- Read header (first non-empty line) to extract column names ---
    with open(filename, "r") as f:
        header = None
        first_line = None
        # Peek the first non-empty line
        for line in f:
            first_line = line.rstrip("\n")
            if first_line.strip():  # non-empty
                break

    # Parse header names if present (expects a line starting with '#')
    header_names: Optional[List[str]] = None
    if first_line is not None and first_line.lstrip().startswith("#"):
        # Strip leading '#', split on whitespace
        header_content = first_line.lstrip()[1:].strip()
        header_names = header_content.split()
    else:
        # No header; we can optionally skip first line if asked, but we don't know names.
        if skip_first and first_line is not None:
            # We will pass skiprows=1 to loadtxt below
            pass
        else:
            # Without a header we cannot reorder by names; raise a clear error
            raise ValueError(
                "No header line starting with '#' found; cannot map names to columns.\n"
                "Add a header like: '# name1 name2 name3 ...' or set skip_first=True and provide a header line."
            )

    # --- Build mapping from requested order -> file columns ---
    name_to_col = {name: idx for idx, name in enumerate(header_names)}
    usecols = []
    names_used = []
    skipped = []

    for name in order_parameters:
        if name in name_to_col:
            usecols.append(name_to_col[name])
            names_used.append(name)
        else:
            skipped.append(name)

    if len(usecols) == 0:
        raise ValueError(
            "None of the requested order_parameters were found in the file header.\n"
            f"Header names: {header_names}"
        )

    # --- Warn about skipped names (e.g., 'const') ---
    if skipped:
        print(f"SKIPPED (not present in file): {', '.join(skipped)}")

    # --- Load the data using the selected columns, in the requested order ---
    # Since we already consumed the first line when probing, reload via loadtxt
    # and tell it that lines starting with '#' are comments.
    # The order of columns in `usecols` determines the output column order.
    data = np.loadtxt(
        filename,
        comments="#",
        usecols=usecols
    )

    # Ensure 2D shape even if there's a single row in the file
    if data.ndim == 1:
        data = data[None, :]

    return (data, names_used) if return_names else data
