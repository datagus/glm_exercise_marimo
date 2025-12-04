# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==6.0.0",
#     "anthropic==0.75.0",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "ucimlrepo==0.0.7",
# ]
# ///

import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    return mo, pd


@app.cell
def _(mo):
    mo.md(text="## Data Fetching")
    return


@app.cell
def _(mo, pd):
    @mo.cache()
    def fetch_bike_data():
        bike_df = pd.read_csv("data/SeoulBikeData.csv", encoding='latin-1')
        return bike_df

    bike_df = fetch_bike_data()
    bike_df
    return


if __name__ == "__main__":
    app.run()
