import datetime as dt
import os.path

import numpy as np
import pandas as pd


def plague():
    """
    Eyam plague dataset from Xu 2016 https://digital.lib.washington.edu/researchworks/handle/1773/37251

    N = 261
    alpha = 3.204
    beta = 0.019 * N = 4.959
    LL = -40.58193 (value from https://rdrr.io/cran/MultiBD/man/dbd_prob.html)
    """

    return np.array([[0., 254, 7, 0],
                     [0.5, 235, 14, 12],
                     [1., 201, 22, 38],
                     [1.5, 153, 29, 79],
                     [2., 121, 20, 120],
                     [2.5, 110, 8, 143],
                     [3., 97, 8, 156],
                     [4., 83, 0, 178]])


def covid19_austria_daily(start=None, end=None, cache_path=None):
    """
    Load Covid-19 data of Austria as a 2D array with columns t, S, I, R. If no cache file is found, the data is
    downloaded from https://www.data.gv.at/katalog/dataset/ef8e980b-9644-45d8-b0e9-c6aaf0eff0c0 and cached as csv.

    Assumes whole population N = 8932664.

    :param start: First date in the data set (e.g. "2020-03-01")
    :param end: Last date in the data set (e.g. "2020-09-01")
    :param cache_path: Path where the csv data should be cached
    :return: SIR data array
    """

    N = 8932664

    if cache_path is not None and os.path.exists(cache_path):

        df = pd.read_csv(cache_path, parse_dates=["Time"]).set_index("Time")

    else:

        # API Docs: https://www.data.gv.at/katalog/dataset/ef8e980b-9644-45d8-b0e9-c6aaf0eff0c0

        # download data
        df = pd.read_csv("https://covid19-dashboard.ages.at/data/CovidFaelle_Timeline.csv", sep=";",
                         parse_dates=["Time"], date_parser=lambda s: dt.datetime.strptime(s, "%d.%m.%Y %H:%M:%S"))
        df = df[df["Bundesland"] == "Österreich"].set_index("Time")
        df = df[["AnzahlFaelleSum", "AnzahlTotSum", "AnzahlGeheiltSum"]]
        df.to_csv(cache_path)

    # calculate SIR
    SIR_df = pd.DataFrame({
        "S": N - df["AnzahlFaelleSum"],
        "I": df["AnzahlFaelleSum"] - df["AnzahlGeheiltSum"] - df["AnzahlTotSum"],
        "R": df["AnzahlGeheiltSum"] + df["AnzahlTotSum"]
    })

    # select range
    SIR_df = SIR_df.loc[slice(start, end)]
    SIR_df["t"] = (SIR_df.index - SIR_df.index.min()).days

    data = SIR_df[["t", "S", "I", "R"]]

    return data.values
