# Bangalore Restaurant Cost Predictor -- Streamlit App

An optional demo front-end for the **Bangalore Restaurant Pricing and Ratings
Analysis** project (Zomato 2022 data, by Joyti Das).

## What the app does

| Tab | Content |
|-----|---------|
| **Predict** | Enter restaurant attributes in the sidebar and get an estimated cost for two in Rs., with the model's typical error noted. |
| **KPIs** | The key performance indicator table computed from the cleaned dataset (matches Section 6 of the notebook). |
| **Explore** | Four charts: restaurants per area, median cost with vs without indoor seating, cuisine popularity, cost distribution. |
| **About** | Dataset source, ODbL licence link, model hyperparameters, test-set metrics (with a sanity check vs the notebook), and a note on limitations including that indoor seating is an association, not a cause. |

The model is a **HistGradientBoostingRegressor** pipeline trained at app start-up
(cached so it only trains once per session).  It uses the same features, preprocessing,
best hyperparameters from the notebook's `RandomizedSearchCV`, target
`log1p(AverageCost)`, 80/20 split, and `random_state=42` as Sections 7 and 9 of
`JoytiDas_BangaloreRestaurantAnalysis.ipynb`.

Expected test-set metrics (random split): R2 ~ 0.666, MAE ~ Rs. 102, RMSE ~ Rs. 195.

## How to run

From the **project root folder** (the folder that contains `data/` and the notebook):

```bash
streamlit run app/app.py
```

The app resolves `data/zomato_cleaned.csv` relative to `app.py`, so this works
locally and on Streamlit Cloud or any other host where the repo is cloned intact.

## Requirements

The app uses only packages already listed in the project's main `requirements.txt`
plus Streamlit itself.  To install into the project `.venv`:

```bash
.venv\Scripts\pip install -r app/requirements.txt
```

Tested with: streamlit 1.64.0, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1,
matplotlib 3.11.2.

## Files

```
app/
  app.py            -- the Streamlit application
  requirements.txt  -- app dependencies (no version pins; see comment for tested versions)
  README.md         -- this file
```

## Submission note

This `app/` folder is **not part of the graded submission**.  The notebook
(`JoytiDas_BangaloreRestaurantAnalysis.ipynb`), report, and cleaned data are
unchanged.  This app is provided as an optional interactive companion for
visitors to the GitHub repository.
