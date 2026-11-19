import numpy as np
import pandas as pd


STATES = ["Low", "Medium", "High"]
EARLY = "2001-2010 within-period"
BOUNDARY = "2010-2011 boundary"
LATE = "2011-2020 within-period"
FULL = "Full period, 2001-2020"


def assign_states(scores, thresholds):
    low, high = thresholds
    return np.select([scores <= low, scores <= high], STATES[:2], default=STATES[2])


def annual_transitions(panel):
    data = panel[["province", "year", "ffd"]].dropna().sort_values(["province", "year"]).copy()
    if data.duplicated(["province", "year"]).any():
        raise ValueError("Each province-year must have one FFD value")
    thresholds = data.ffd.quantile([0.33, 0.67])
    data["state"] = assign_states(data.ffd, thresholds)
    following = data.groupby("province")[["year", "state"]].shift(-1)
    consecutive = following.year.eq(data.year + 1)
    events = data.loc[consecutive, ["province", "year", "state"]].rename(
        columns={"year": "origin_year", "state": "origin_state"}
    )
    events["destination_year"] = following.loc[consecutive, "year"].astype(int)
    events["destination_state"] = following.loc[consecutive, "state"]
    events["period"] = np.select(
        [events.origin_year < 2010, events.origin_year == 2010],
        [EARLY, BOUNDARY], default=LATE,
    )

    matrices = []
    for period in [FULL, EARLY, BOUNDARY, LATE]:
        subset = events if period == FULL else events.loc[events.period.eq(period)]
        counts = pd.crosstab(subset.origin_state, subset.destination_state).reindex(
            index=STATES, columns=STATES, fill_value=0
        )
        totals = counts.sum(axis=1)
        probabilities = counts.div(totals.replace(0, np.nan), axis=0).fillna(0)
        for origin in STATES:
            for destination in STATES:
                count = counts.at[origin, destination]
                share = count / len(events) if period == FULL and len(events) else np.nan
                matrices.append((period, origin, destination, count, totals[origin],
                                 probabilities.at[origin, destination], share, len(subset)))
    counts = pd.DataFrame(matrices, columns=[
        "period", "origin_state", "destination_state", "count", "origin_state_total",
        "conditional_probability", "share_of_all_full_period_transitions", "period_transition_total",
    ])
    yearly = pd.crosstab(data.year, data.state).reindex(columns=STATES, fill_value=0)
    return events, counts, yearly.div(yearly.sum(axis=1), axis=0).reset_index(), thresholds
