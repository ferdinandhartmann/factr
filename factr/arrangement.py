import numpy as np


def arrangement_id_to_one_hot(raw_arrangement):
    """Decode a base-3 arrangement ID into three concatenated height one-hots."""
    values = np.asarray(raw_arrangement).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected one arrangement ID, got {values.size} values.")
    value = float(values[0])
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"Arrangement ID must be a finite integer, got {values[0]!r}.")

    arrangement_id = int(value)
    if arrangement_id < 0 or arrangement_id > 26:
        raise ValueError(f"Arrangement ID must be in [0, 26], got {arrangement_id}.")

    # Goal 3 is the least-significant base-3 digit in the spreadsheet IDs.
    goal_1 = arrangement_id // 9
    goal_2 = (arrangement_id // 3) % 3
    goal_3 = arrangement_id % 3
    return np.eye(3, dtype=np.float32)[[goal_1, goal_2, goal_3]].reshape(9)