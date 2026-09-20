# =============================================================================
# Bangalore Restaurant Pricing and Ratings Analysis -- Streamlit Demo App
# Project by Joyti Das (AICTE | IBM SkillsBuild Data Analytics Internship 2026)
#
# Run from the project root with:
#   streamlit run app/app.py
# =============================================================================

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# CONFIG -- mirrors the notebook's CONFIG cell exactly
# ---------------------------------------------------------------------------
RANDOM_STATE       = 42
TEST_SIZE          = 0.20
TOP_N_AREAS        = 30
CHAIN_MIN_OUTLETS  = 5

# Best hyperparameters found by RandomizedSearchCV in Section 9 of the notebook
# (loguniform(0.01,0.3) x randint(100,500) x randint(3,8), n_iter=20, cv=5)
HGB_LEARNING_RATE  = 0.03574712922600244
HGB_MAX_ITER       = 370
HGB_MAX_DEPTH      = 7

# Notebook test-set metrics (random split) -- used in the sanity check
NB_R2   = 0.6664
NB_MAE  = 101.8   # Rs.
NB_RMSE = 195.4   # Rs.

# ---------------------------------------------------------------------------
# Data path -- resolved relative to THIS file so it works wherever you run it
# ---------------------------------------------------------------------------
DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "zomato_cleaned.csv"

# ---------------------------------------------------------------------------
# Top 30 areas (notebook Section 7, in value_counts order)
# ---------------------------------------------------------------------------
TOP_AREAS_LIST = [
    "Electronic City", "Marathahalli", "HSR", "Whitefield", "BTM",
    "Indiranagar", "JP Nagar", "Sarjapur Road", "Rajajinagar", "New BEL Road",
    "Banashankari", "Jayanagar", "Kalyan Nagar", "Bellandur", "Bannerghatta Road",
    "Brookefield", "Kammanahalli", "Bommanahalli", "Malleshwaram", "Basavanagudi",
    "Vijay Nagar", "Yeshwantpur", "Koramangala 5th Block", "Banaswadi",
    "Frazer Town", "KR Puram", "RT Nagar", "Nagawara",
    "Basaveshwara Nagar", "Koramangala 1st Block",
]

# Top 20 cuisines (notebook Section 5 Step 8, in value_counts order)
TOP_CUISINES_LIST = [
    "North Indian", "Beverages", "Chinese", "Desserts", "Fast Food",
    "Biryani", "South Indian", "Street Food", "Shake", "Mughlai",
    "Ice Cream", "Bakery", "Pizza", "Burger", "Rolls",
    "Seafood", "Kebab", "Sandwich", "Momos", "Andhra",
]

# Cuisine column names in the CSV (cui_<Name_with_underscores>)
CUI_COLS = [
    "cui_North_Indian", "cui_Beverages", "cui_Chinese", "cui_Desserts",
    "cui_Fast_Food", "cui_Biryani", "cui_South_Indian", "cui_Street_Food",
    "cui_Shake", "cui_Mughlai", "cui_Ice_Cream", "cui_Bakery",
    "cui_Pizza", "cui_Burger", "cui_Rolls", "cui_Seafood",
    "cui_Kebab", "cui_Sandwich", "cui_Momos", "cui_Andhra",
]

FLAG_COLS    = ["IsHomeDelivery", "isTakeaway", "isIndoorSeating", "isVegOnly"]
NUMERIC_COLS = ["NumCuisines", "IsChain"] + FLAG_COLS + CUI_COLS
CAT_COLS     = ["AreaGroup"]
FEATURE_COLS = CAT_COLS + NUMERIC_COLS


# ===========================================================================
# Data loading
# ===========================================================================
@st.cache_data
def load_data():
    """Load the cleaned CSV and recompute the three engineered columns
    exactly as Section 7 of the notebook does:
      - NumCuisines  : number of cuisines in CuisineListStr
      - IsChain      : 1 if restaurant name appears >= CHAIN_MIN_OUTLETS times
      - AreaGroup    : Area if in top-30, else 'Other'
    """
    df = pd.read_csv(DATA_PATH)

    # NumCuisines: count pipe-separated items in CuisineListStr
    df["NumCuisines"] = (
        df["CuisineListStr"]
        .fillna("")
        .str.split("|")
        .apply(lambda items: len([x for x in items if x.strip()]))
    )

    # IsChain: same threshold as notebook (CHAIN_MIN_OUTLETS = 5)
    name_counts = df["Name"].value_counts()
    df["IsChain"] = df["Name"].map(name_counts).ge(CHAIN_MIN_OUTLETS).astype(int)

    # AreaGroup: top-30 areas by name; rest become 'Other'
    top_areas = df["Area"].value_counts().head(TOP_N_AREAS).index.tolist()
    df["AreaGroup"] = df["Area"].where(df["Area"].isin(top_areas), other="Other")

    return df


# ===========================================================================
# Model training
# ===========================================================================
@st.cache_resource
def train_model(df):
    """Train the HistGradientBoosting pipeline on log1p(AverageCost) using the
    same 80/20 random split and hard-coded best hyperparameters from the
    notebook's RandomizedSearchCV (Section 9).

    Returns the fitted pipeline and the test-set metrics dict.
    """
    # Target: log1p(AverageCost) -- same as notebook Section 7
    df["LogAverageCost"] = np.log1p(df["AverageCost"])

    X = df[FEATURE_COLS].copy()
    y = df["LogAverageCost"].copy()

    # Same split as notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Preprocessor: OrdinalEncoder for AreaGroup (safe for trees) + StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT_COLS,
            ),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )

    # Pipeline with hard-coded best hyperparameters from the notebook search
    pipe = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=HGB_LEARNING_RATE,
                    max_iter=HGB_MAX_ITER,
                    max_depth=HGB_MAX_DEPTH,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    # Evaluate on held-out test set (log scale then convert back to Rs.)
    y_pred = pipe.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    rmse = float(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred))))

    metrics = {"r2": round(r2, 4), "mae": round(mae, 1), "rmse": round(rmse, 1)}
    return pipe, metrics


# ===========================================================================
# KPI helper
# ===========================================================================
@st.cache_data
def compute_kpis(df):
    """Compute the KPI table shown in the KPIs tab (cleaned data values)."""
    total = len(df)
    median_cost = df["AverageCost"].median()
    mean_cost   = df["AverageCost"].mean()

    del_rated = df["Delivery Ratings"].notna()
    mean_del  = df["Delivery Ratings"].mean()
    share_gte4 = (df["Delivery Ratings"] >= 4.0).sum() / del_rated.sum() * 100
    del_cov    = del_rated.sum() / total * 100
    din_cov    = df["Dinner Ratings"].notna().sum() / total * 100

    home_del  = df["IsHomeDelivery"].mean() * 100
    takeaway  = df["isTakeaway"].mean() * 100
    indoor    = df["isIndoorSeating"].mean() * 100
    veg_only  = df["isVegOnly"].mean() * 100

    name_counts = df["Name"].value_counts()
    chain_names = (name_counts >= CHAIN_MIN_OUTLETS).sum()
    chain_rows  = name_counts[name_counts >= CHAIN_MIN_OUTLETS].sum()

    rows = [
        ("K1",  "Total restaurants",               f"{total:,}"),
        ("K2",  "Median AverageCost (Rs.)",         f"{median_cost:.0f}"),
        ("K3",  "Mean AverageCost (Rs.)",           f"{mean_cost:.1f}"),
        ("K4",  "Mean Delivery Rating (rated)",     f"{mean_del:.3f}"),
        ("K5",  "Rated restaurants (delivery)",     f"{del_rated.sum():,}"),
        ("K6",  "Share delivery rated >= 4.0",      f"{share_gte4:.1f}%"),
        ("K7",  "Delivery rating coverage",         f"{del_cov:.1f}%"),
        ("K8",  "Dinner rating coverage",           f"{din_cov:.1f}%"),
        ("K11", "Home delivery (%)",                f"{home_del:.1f}%"),
        ("K11", "Takeaway (%)",                     f"{takeaway:.1f}%"),
        ("K11", "Indoor seating (%)",               f"{indoor:.1f}%"),
        ("K11", "Veg-only (%)",                     f"{veg_only:.1f}%"),
        ("K12", "Chain names (>= 5 outlets)",       f"{chain_names}"),
        ("K12", "Chain restaurant rows",            f"{chain_rows:,}"),
    ]
    return pd.DataFrame(rows, columns=["KPI", "Description", "Value"])


# ===========================================================================
# Prediction helper
# ===========================================================================
def build_input_row(area, cuisines, home_del, takeaway, indoor, veg_only, is_chain):
    """Convert sidebar selections into a single-row DataFrame matching FEATURE_COLS."""
    # AreaGroup
    area_group = area if area in TOP_AREAS_LIST else "Other"

    # NumCuisines: at least 1 (every real restaurant has at least one cuisine)
    num_cuisines = max(len(cuisines), 1)

    # IsChain
    chain_flag = 1 if is_chain else 0

    # Service flags
    flags = {
        "IsHomeDelivery": int(home_del),
        "isTakeaway":     int(takeaway),
        "isIndoorSeating": int(indoor),
        "isVegOnly":      int(veg_only),
    }

    # Cuisine dummies: 1 if that cuisine was selected
    cui_map = {
        "North Indian":  "cui_North_Indian",
        "Beverages":     "cui_Beverages",
        "Chinese":       "cui_Chinese",
        "Desserts":      "cui_Desserts",
        "Fast Food":     "cui_Fast_Food",
        "Biryani":       "cui_Biryani",
        "South Indian":  "cui_South_Indian",
        "Street Food":   "cui_Street_Food",
        "Shake":         "cui_Shake",
        "Mughlai":       "cui_Mughlai",
        "Ice Cream":     "cui_Ice_Cream",
        "Bakery":        "cui_Bakery",
        "Pizza":         "cui_Pizza",
        "Burger":        "cui_Burger",
        "Rolls":         "cui_Rolls",
        "Seafood":       "cui_Seafood",
        "Kebab":         "cui_Kebab",
        "Sandwich":      "cui_Sandwich",
        "Momos":         "cui_Momos",
        "Andhra":        "cui_Andhra",
    }
    cui_vals = {col: (1 if name in cuisines else 0) for name, col in cui_map.items()}

    row = {
        "AreaGroup":   area_group,
        "NumCuisines": num_cuisines,
        "IsChain":     chain_flag,
        **flags,
        **cui_vals,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ===========================================================================
# Charts for the Explore tab
# ===========================================================================
@st.cache_data
def make_explore_charts(df):
    """Return four Matplotlib figures for the Explore tab."""
    figs = []

    # -- Chart 1: Restaurants per area (top 15) --------------------------------
    area_counts = df["Area"].value_counts().head(15)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    area_counts.sort_values().plot(kind="barh", ax=ax1, color="#4472C4")
    ax1.set_title("Restaurants per Area (top 15)")
    ax1.set_xlabel("Number of restaurants")
    ax1.set_ylabel("Area")
    plt.tight_layout()
    figs.append(fig1)

    # -- Chart 2: Median cost -- with vs without indoor seating ----------------
    indoor_cost = (
        df.groupby("isIndoorSeating")["AverageCost"]
        .median()
        .rename({0: "No indoor seating", 1: "Indoor seating"})
    )
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    indoor_cost.plot(kind="bar", ax=ax2, color=["#ED7D31", "#4472C4"], rot=0)
    ax2.set_title("Median Cost for Two (Rs.) -- Indoor Seating")
    ax2.set_ylabel("Median AverageCost (Rs.)")
    ax2.set_xlabel("")
    for bar in ax2.patches:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"Rs. {bar.get_height():.0f}",
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    figs.append(fig2)

    # -- Chart 3: Cuisine popularity (top 15 by restaurant count) --------------
    cui_sums = df[CUI_COLS].sum().sort_values(ascending=False).head(15)
    cui_sums.index = [c.replace("cui_", "").replace("_", " ") for c in cui_sums.index]
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    cui_sums.sort_values().plot(kind="barh", ax=ax3, color="#70AD47")
    ax3.set_title("Cuisine Popularity (restaurants serving each -- top 15)")
    ax3.set_xlabel("Number of restaurants")
    ax3.set_ylabel("Cuisine")
    plt.tight_layout()
    figs.append(fig3)

    # -- Chart 4: AverageCost distribution (capped at 99th percentile) ---------
    cap = df["AverageCost"].quantile(0.99)
    cost_capped = df["AverageCost"].clip(upper=cap)
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.hist(cost_capped, bins=40, color="#4472C4", edgecolor="white")
    ax4.axvline(df["AverageCost"].median(), color="#ED7D31", linewidth=2,
                label=f"Median Rs. {df['AverageCost'].median():.0f}")
    ax4.axvline(df["AverageCost"].mean(), color="#FF0000", linewidth=2, linestyle="--",
                label=f"Mean Rs. {df['AverageCost'].mean():.0f}")
    ax4.set_title(f"AverageCost Distribution (capped at Rs. {cap:.0f} = 99th pct)")
    ax4.set_xlabel("Cost for Two (Rs.)")
    ax4.set_ylabel("Count")
    ax4.legend()
    plt.tight_layout()
    figs.append(fig4)

    return figs


# ===========================================================================
# Page layout
# ===========================================================================
st.set_page_config(
    page_title="Bangalore Restaurant Cost Predictor",
    page_icon=":fork_and_knife:",
    layout="wide",
)

st.title("Bangalore Restaurant Cost Predictor")
st.markdown(
    "Explore pricing patterns in Bangalore's restaurant scene (Zomato 2022 data) "
    "and estimate the cost for two based on area, cuisines and services.  \n"
    "Use the sidebar to set restaurant attributes, then see the prediction in the "
    "**Predict** tab."
)

# ---------------------------------------------------------------------------
# Load data and train model
# ---------------------------------------------------------------------------
df = load_data()
pipe, app_metrics = train_model(df)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Restaurant Attributes")

area_options = ["Other"] + TOP_AREAS_LIST
selected_area = st.sidebar.selectbox("Area", area_options, index=6)  # default: JP Nagar

selected_cuisines = st.sidebar.multiselect(
    "Cuisines (select all that apply)",
    options=TOP_CUISINES_LIST,
    default=["North Indian", "Chinese"],
    help="Only the 20 most common cuisines are available. "
         "Selections map to the cuisine dummy features used in the model.",
)

st.sidebar.markdown("**Services**")
cb_home_del = st.sidebar.checkbox("Home delivery",   value=True)
cb_takeaway = st.sidebar.checkbox("Takeaway",        value=False)
cb_indoor   = st.sidebar.checkbox("Indoor seating",  value=False)
cb_veg_only = st.sidebar.checkbox("Veg only",        value=False)
cb_chain    = st.sidebar.checkbox("Is a chain restaurant", value=False)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_kpis, tab_explore, tab_about = st.tabs(
    ["Predict", "KPIs", "Explore", "About"]
)

# ===========================================================================
# TAB 1 -- PREDICT
# ===========================================================================
with tab_predict:
    st.subheader("Estimated Cost for Two")

    input_row = build_input_row(
        area=selected_area,
        cuisines=selected_cuisines,
        home_del=cb_home_del,
        takeaway=cb_takeaway,
        indoor=cb_indoor,
        veg_only=cb_veg_only,
        is_chain=cb_chain,
    )

    log_pred  = pipe.predict(input_row)[0]
    pred_cost = float(np.expm1(log_pred))

    # Result card
    st.markdown(
        f"""
        <div style="
            background:#f0f4ff;
            border-left:6px solid #4472C4;
            padding:1.2em 1.5em;
            border-radius:6px;
            margin-bottom:1em;
        ">
        <h2 style="margin:0;color:#1a1a2e;">
            Estimated cost for two: &nbsp; <span style="color:#4472C4;">Rs. {pred_cost:,.0f}</span>
        </h2>
        <p style="margin:0.4em 0 0 0;color:#555;font-size:0.95em;">
            Typical error: about Rs. 102 (mean absolute error on test data)
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary of inputs
    with st.expander("Input summary"):
        st.write({
            "Area (AreaGroup)":    selected_area if selected_area in TOP_AREAS_LIST else "Other",
            "Cuisines selected":   selected_cuisines if selected_cuisines else ["(none -- NumCuisines set to 1)"],
            "NumCuisines":         max(len(selected_cuisines), 1),
            "IsChain":             int(cb_chain),
            "IsHomeDelivery":      int(cb_home_del),
            "isTakeaway":          int(cb_takeaway),
            "isIndoorSeating":     int(cb_indoor),
            "isVegOnly":           int(cb_veg_only),
        })

# ===========================================================================
# TAB 2 -- KPIs
# ===========================================================================
with tab_kpis:
    st.subheader("Key Performance Indicators (cleaned dataset)")
    st.markdown(
        "These figures come directly from the cleaned `data/zomato_cleaned.csv` "
        "and match the KPI table in Section 6 of the notebook."
    )
    kpi_df = compute_kpis(df)
    st.dataframe(kpi_df, width="stretch", hide_index=True)

# ===========================================================================
# TAB 3 -- EXPLORE
# ===========================================================================
with tab_explore:
    st.subheader("Explore the Data")
    figs = make_explore_charts(df)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(figs[0])   # restaurants per area
    with col2:
        st.pyplot(figs[1])   # cost indoor vs no indoor

    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(figs[2])   # cuisine popularity
    with col4:
        st.pyplot(figs[3])   # cost distribution

# ===========================================================================
# TAB 4 -- ABOUT
# ===========================================================================
with tab_about:
    st.subheader("About This App")

    st.markdown(
        """
### Dataset
- **Name:** Zomato Bangalore Restaurants 2022
- **Author:** vora1011
- **Source:** [Kaggle](https://www.kaggle.com/datasets/vora1011/zomato-bangalore-restaurants-2022)
- **Collected:** March 2022
- **Licence:** Open Database Licence (ODbL) --
  [https://opendatacommons.org/licenses/odbl/](https://opendatacommons.org/licenses/odbl/)

---

### Model
- **Algorithm:** HistGradientBoostingRegressor (scikit-learn)
- **Target:** log1p(AverageCost) -- predictions are converted back to Rs. with expm1()
- **Features (27 total):**
  - `AreaGroup` -- top 30 areas by restaurant count; anything else becomes "Other"
  - `IsHomeDelivery`, `isTakeaway`, `isIndoorSeating`, `isVegOnly` -- service flags
  - `NumCuisines` -- number of cuisines selected (minimum 1)
  - `IsChain` -- restaurant name appears >= 5 times in the dataset
  - 20 cuisine dummy columns (cui_*) for the 20 most common cuisines
  - Only the 20 most common cuisines can be selected in the sidebar because the
    model was trained with dummies for only those 20.
- **Split:** 80 / 20 random split, random_state=42
- **Hyperparameters:** Hard-coded from the notebook's RandomizedSearchCV result
  (Section 9 of JoytiDas_BangaloreRestaurantAnalysis.ipynb):
  - learning_rate = 0.03574712922600244
  - max_iter = 370
  - max_depth = 7
"""
    )

    # Sanity check: compare app metrics vs notebook reference
    st.markdown("---")
    st.markdown("### Model Metrics (test set, random 80/20 split)")

    tolerance = 0.001   # allow tiny float differences across scikit-learn minor versions
    r2_ok   = abs(app_metrics["r2"]   - NB_R2)   < tolerance
    mae_ok  = abs(app_metrics["mae"]  - NB_MAE)  < 1.0    # within Rs. 1
    rmse_ok = abs(app_metrics["rmse"] - NB_RMSE) < 1.0

    metrics_df = pd.DataFrame(
        {
            "Metric": ["R2", "MAE (Rs.)", "RMSE (Rs.)"],
            "App (this run)": [
                app_metrics["r2"],
                app_metrics["mae"],
                app_metrics["rmse"],
            ],
            "Notebook reference": [NB_R2, NB_MAE, NB_RMSE],
            "Match?": [
                "Yes" if r2_ok   else "DIFFERS",
                "Yes" if mae_ok  else "DIFFERS",
                "Yes" if rmse_ok else "DIFFERS",
            ],
        }
    )
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    if not all([r2_ok, mae_ok, rmse_ok]):
        st.warning(
            "One or more metrics differ from the notebook reference. "
            "This can happen when a different version of scikit-learn or numpy "
            "changes internal tie-breaking or rounding in "
            "HistGradientBoostingRegressor. The model is still trained with the "
            "same data, split and hyperparameters."
        )
    else:
        st.success("App metrics match the notebook reference.")

    st.markdown(
        """
---

### Limitations
1. **Association, not causation.** Indoor seating is the strongest model feature
   (permutation importance 0.84). This reflects an *association* in the data --
   restaurants with indoor seating tend to cost more -- not a causal mechanism.
   Fitting out a takeaway counter with tables would not automatically raise prices.
2. **Scope.** The model was trained on Bangalore restaurants scraped in March 2022.
   It may not generalise to other cities or later time periods.
3. **Rating columns excluded.** Dinner and Delivery Ratings were not used as features
   because predicting *cost* from ratings could introduce endogeneity, and
   rating coverage was uneven (Dinner Ratings 40 % coverage).
4. **Review counts excluded.** Review counts are endogenous -- a restaurant must
   already be open and priced before it accumulates reviews.
5. **Moderate accuracy.** R2 ~ 0.67 means the model explains about two-thirds of
   the variance in log-cost. The remaining third is driven by factors not in this
   dataset (menu items, decor, brand premium, etc.).
"""
    )
