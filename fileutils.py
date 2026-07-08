"""
Utility module — data I/O and re-exports.

Re-exports the public API from mesher, FEmodel, and viewer sub-modules
so that notebooks can ``import utils`` as before.
"""
import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def commas_to_dots(x):
    """Replace commas with dots and encode to bytes (for European decimal format)."""
    return x.replace(',', '.').encode()


def extract_data_from_tensile_test(file, segment_number='all', tare=False):
    """
    Extract data from a tensile test CSV file.

    The CSV is expected to have columns (semicolon-delimited):
      col 0: displacement [mm]
      col 1: force [N]
      col 3: time [s]
      col 5: segment number
    """
    try:
        data = np.genfromtxt(
            (commas_to_dots(x) for x in open(file, encoding='iso-8859-1')),
            delimiter=';', skip_header=2
        )
    except Exception as e:
        raise IOError(f"Failed to read tensile test file '{file}': {e}")

    if data.ndim < 1 or data.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns in '{file}', got shape {data.shape}. "
            "Check the CSV format."
        )

    displacement = data[:, 0]
    force = data[:, 1]
    time = data[:, 3]
    segment = data[:, 5].copy()

    prev_seg = segment[0]
    seg_id = 0
    for i in range(len(segment)):
        if segment[i] != prev_seg:
            prev_seg = segment[i]
            seg_id += 1
        segment[i] = seg_id

    if segment_number != 'all':
        mask = segment == segment_number
        if not np.any(mask):
            raise ValueError(
                f"Segment {segment_number} not found. "
                f"Available: {list(np.unique(segment))}"
            )
        displacement = displacement[mask]
        force = force[mask]
        time = time[mask]

    if tare:
        force -= force[0]
        time -= time[0]
        displacement -= displacement[0]

    return displacement, force, time


def read_mat_data(file):
    """
    Read a MATLAB .mat file and clean it up.

    - Removes all-zero rows/columns
    - Replaces 0 with NaN
    """
    data = scipy.io.loadmat(file)
    for field in list(data.keys()):
        arr = data[field]
        if not isinstance(arr, np.ndarray) or arr.dtype.kind not in ('f', 'i'):
            continue
        arr = arr[~np.all(arr == 0, axis=1)]
        arr = arr[:, ~np.all(arr == 0, axis=0)]
        arr[arr == 0] = np.nan
        data[field] = arr
    return data
