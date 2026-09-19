# Bangalore Restaurant Pricing and Ratings Analysis (Zomato, 2022)

**Student:** Joyti Das  
**Programme:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares)  
**GitHub:** https://github.com/JOYTI-DAS/Zomato-Bangalore-Restaurant-Analysis

---

## Problem Statement

What drives restaurant pricing and customer ratings in Bangalore, and can we predict the cost for two (AverageCost) from a restaurant's area, cuisines and services?

---

## Dataset

| Item | Detail |
|---|---|
| Source | Kaggle — "Zomato Bangalore Restaurants 2022" |
| Author | vora1011 |
| URL | https://www.kaggle.com/datasets/vora1011/zomato-bangalore-restaurants-2022 |
| Collected | March 2022 |
| License | Open Database License (ODbL) v1.0 - https://opendatacommons.org/licenses/odbl/1-0/ |
| Raw size | 8,923 rows × 19 columns |
| Local file | `BangaloreZomatoData.csv` (read-only — never modified) |

### Columns kept after cleaning (12 original + 20 cuisine dummies + 1 list string = 33 total)

| Column | Description |
|---|---|
| Name | Restaurant name |
| Cuisines | Raw comma-separated cuisine string |
| Area | Area name (", Bangalore" suffix stripped) |
| IsHomeDelivery | 1 = offers home delivery |
| isTakeaway | 1 = offers takeaway |
| isIndoorSeating | 1 = has indoor seating |
| isVegOnly | 1 = vegetarian-only menu |
| Dinner Ratings | Dine-in rating (NaN where missing; never imputed) |
| Dinner Reviews | Dine-in review count |
| Delivery Ratings | Delivery rating (NaN where missing; never imputed) |
| Delivery Reviews | Delivery review count |
| AverageCost | Cost for two in rupees (model target) |
| cui_* (20 cols) | Binary flag: 1 if restaurant serves that cuisine |
| CuisineListStr | Pipe-separated cuisine list for CSV storage |

### Columns dropped

| Column | Reason |
|---|---|
| URL, PhoneNumber, Full_Address | PII / not useful for analysis |
| Timing | 34.78% missing; scrape-time dependent |
| KnownFor | 97.11% missing |
| PopularDishes | 82.80% missing |
| PeopleKnownFor | 60.95% missing |

---

## Cleaning Summary

- `"-"` in rating columns replaced with NaN (never imputed)
- `", Bangalore"` suffix stripped from all Area values
- Two restaurant names (`Art of Delight`, `Belgian Waffle Factory`) gained trailing-space variants merged after `.str.strip()`
- AverageCost: no rows deleted; `log1p` transform used for modelling; chart display capped at 99th percentile (Rs 1,600)
- 0 fully duplicated rows; 0 rows dropped for missing AverageCost

---

## Technologies Used

| Library | Version | Purpose |
|---|---|---|
| pandas | 3.0.6 | Data loading, cleaning, KPIs |
| numpy | 2.4.6 | Numerical operations, log transforms |
| matplotlib | 3.11.2 | Charts |
| seaborn | 0.13.2 | Statistical plots |
| scikit-learn | 1.9.1 | Pipelines, encoding, models, metrics |
| scipy | 1.17.1 | Distribution sampling for hyperparameter search |
| python-docx | 1.2.0 | Project report generation |
| jupyter | 1.1.1 | Notebook environment |
| nbformat | 5.11.1 | Notebook format |
| nbconvert | 7.17.1 | Notebook execution and export |
| ipykernel | 7.3.0 | Jupyter kernel |

---

## Project Structure

```
IBM FINAL PROJECT/
├── BangaloreZomatoData.csv          # Raw dataset (read-only)
├── JoytiDas_BangaloreRestaurantAnalysis.ipynb  # Main notebook (run top-to-bottom)
├── requirements.txt                 # Pinned library versions
├── README.md                        # This file
├── JoytiDas_ProjectReport.docx      # Full project report
├── PROJECT_WALKTHROUGH.md           # Plain-language study guide
├── data/
│   └── zomato_cleaned.csv           # Cleaned dataset (8,923 rows × 33 cols)
└── figures/
    ├── eda_Q1.png  …  eda_Q10.png   # EDA charts
    ├── model_importance.png          # Permutation importance
    ├── model_residuals.png           # Residual plot
    └── model_confusion.png           # Classification confusion matrix
```

---

## Setup and Run Instructions

```bash
# 1. Create a virtual environment inside the project folder
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the kernel (so Jupyter uses the venv)
python -m ipykernel install --user --name=zomato_venv

# 5. Open the notebook
jupyter notebook JoytiDas_BangaloreRestaurantAnalysis.ipynb

# 6. Run All (Kernel > Restart & Run All)
```

> The notebook must be run **top to bottom** in order. All random operations use `RANDOM_STATE=42`.

---

## KPI Summary Table

All values computed from executed notebook output.

| KPI | Value |
|---|---|
| Total restaurants | 8,923 |
| Median AverageCost | Rs 250 |
| Mean AverageCost | Rs 340 |
| Mean Delivery Rating (rated only) | 3.879 |
| Rated restaurants (delivery) | 7,792 |
| Share delivery rated ≥ 4.0 | 48.7% |
| Delivery rating coverage | 87.3% |
| Dinner (dine-in) rating coverage | 40.3% |
| Home delivery share | 99.8% |
| Takeaway share | 66.0% |
| Indoor seating share | 44.3% |
| Veg-only share | 7.2% |
| Chain names (≥ 5 outlets) | 296 |
| Chain restaurant rows | 3,007 |
| AverageCost 99th percentile | Rs 1,600 |
| AverageCost maximum | Rs 4,200 |

---

## Key Results and Insights

1. **Pricing is right-skewed.** Median Rs 250, mean Rs 340, max Rs 4,200. Most restaurants are budget-segment.
2. **Indoor seating is the strongest cost predictor.** Restaurants with indoor seating have a median cost of Rs 400 vs Rs 200 for delivery/takeaway-only (2× difference). Removing `isIndoorSeating` drops model R² from 0.664 to 0.474. This is a strong **association**, not a causal claim.
3. **Area (neighbourhood) matters.** AreaGroup is the 9th most important feature by permutation importance; Electronic City has the most restaurants (674), while Koramangala 4th Block has the highest median cost (Rs 375).
4. **Chains and independents share the same median price (Rs 250).** However, the independent mean (Rs 356) exceeds chains (Rs 309), which may reflect a minority of premium independents.
5. **Delivery ratings are nearly unpredictable** from structural features. Pearson r (rating vs log-cost) = 0.050. Classification model (predict rating ≥ 4.0) reaches only 63.6% accuracy vs 51.3% majority-class baseline.
6. **North Indian (3,249 restaurants) and Biryani (1,987)** dominate — high-competition segments. Seafood has the highest median cost per cuisine (Rs 500).
7. **Zomato Bangalore 2022 is a delivery-first market:** 99.8% of restaurants offer home delivery; only 44.3% have indoor seating.

---

## Best Regression Model Summary

| Split | Model | R² | MAE (Rs) | RMSE (Rs) |
|---|---|---|---|---|
| Random 80/20 | HistGradientBoosting | **0.666** | Rs 102 | Rs 195 |
| Group (by Name) | HistGradientBoosting | **0.622** | Rs 111 | Rs 212 |
| Random 80/20 | Ridge (one-hot) | 0.592 | Rs 116 | Rs 218 |
| Random 80/20 | Random Forest | 0.658 | Rs 101 | Rs 195 |
| Random 80/20 | Median baseline | -0.001 | Rs 175 | Rs 319 |

Encoder: Ridge uses OneHotEncoder for AreaGroup (correct for linear models); RF and HGB use OrdinalEncoder (safe for tree models). All transforms fitted inside sklearn Pipeline on training data only (no data leakage).

---

## Parameters Summary

| Parameter | Value |
|---|---|
| RANDOM_STATE | 42 |
| TEST_SIZE | 0.20 |
| CV_FOLDS | 5 |
| RATING_THRESHOLD | 4.0 |
| OUTLIER_PERCENTILE | 99 (chart cap Rs 1,600) |
| TOP_N_CUISINES | 20 |
| TOP_N_AREAS | 30 |
| CHAIN_MIN_OUTLETS | 5 |
| HGB best alpha / params | learning_rate≈0.036, max_depth=7, max_iter=370 |

---

## Limitations

- Data from March 2022 only — a static snapshot.
- Scraped listing data; self-reported by restaurant owners.
- 59.7% of dine-in ratings are missing (not imputed).
- All findings are correlations — no causal claims are made.
- Chain repetition: same chain in train and test with random split (group split provided as honest alternative).
- Timing column dropped (34.78% missing, scrape-time dependent).

---

## Author

**Joyti Das**  
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares)  
GitHub: [TODO: add your GitHub URL here]

---

## References

- Dataset: vora1011, "Zomato Bangalore Restaurants 2022", Kaggle, 2022. https://www.kaggle.com/datasets/vora1011/zomato-bangalore-restaurants-2022
- scikit-learn documentation: https://scikit-learn.org/stable/
- pandas documentation: https://pandas.pydata.org/docs/
