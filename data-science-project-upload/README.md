# King County House Price Prediction

## 1. Problem / Research Question

**Scenario:** You are a Data Analyst working at a Real Estate Investment Trust (REIT). The Trust wants to start investing in residential real estate and needs to determine the market price of a house given a set of attributes.

**Research question:** How accurately can we predict the sale price of a house in King County (which includes Seattle, WA) using features such as square footage, number of bedrooms and bathrooms, number of floors, waterfront view, condition, grade, and location coordinates — and which modelling approach (linear, polynomial, or regularised regression) gives the best out-of-sample performance?

---

## 2. Dataset Source and Usage Permissions

- **Dataset:** *House Sales in King County, USA*
- **Source:** Kaggle — https://www.kaggle.com/harlfoxem/housesalesprediction
- **File used:** `kc_house_data.csv` (21,613 home sales)
- **Time period:** Houses sold between **May 2014 and May 2015**
- **License / usage:** The dataset is publicly available on Kaggle for educational and machine-learning practice. It contains no personally identifiable information about buyers or sellers; records are keyed by an anonymised house `id`.
- **Permissions note:** No passwords, API keys, or personal data are committed to this repository. The raw CSV is not stored here; instructions below tell you where to download it.

### Key variables
| Variable | Description |
|---|---|
| `price` | Sale price (target, USD) |
| `bedrooms`, `bathrooms` | Count of bedrooms / bathrooms |
| `sqft_living`, `sqft_above`, `sqft_basement` | Square footage (living / above-ground / basement) |
| `floors` | Number of floors |
| `waterfront` | Binary — whether the house has a waterfront view |
| `view`, `condition`, `grade` | Quality / condition ratings |
| `yr_built`, `yr_renovated` | Construction / renovation year |
| `zipcode`, `lat`, `long` | Location |

---

## 3. Methods and Tools

### Tools
- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualisation:** `matplotlib`, `seaborn` (boxplot, regplot)
- **Modelling:** `scikit-learn` — `LinearRegression`, `Ridge`, `StandardScaler`, `PolynomialFeatures`, `Pipeline`, `train_test_split`
- **Environment:** Jupyter Notebook

### Methods
1. **Data wrangling:** Dropped the `id` and `Unnamed: 0` columns; inspected and imputed missing values in `bedrooms` and `bathrooms`.
2. **Exploratory Data Analysis (EDA):**
   - `describe()` summary statistics.
   - Seaborn **boxplot** comparing price distributions for houses with vs. without a waterfront view (to identify outliers).
   - Seaborn **regplot** examining the relationship between `sqft_above` and `price`.
3. **Model development:**
   - Simple linear regression: `sqft_living` → `price`.
   - Multiple linear regression using a feature list (`features`).
   - Scikit-learn **Pipeline** = `StandardScaler` → `PolynomialFeatures(include_bias=False)` → `LinearRegression`.
4. **Model evaluation & refinement:**
   - Train/test split.
   - **Ridge regression** with `alpha = 0.1` on the test set.
   - **2nd-order polynomial Ridge** (`PolynomialFeatures(degree=2)`) with `alpha = 0.1` on the test set.
   - Compared R² across all models.

---

## 4. Key Findings and One Limitation

### Key findings (measured R² from the notebook)
| Model | R² |
|---|---|
| Linear regression, `sqft_living` only | **0.493** |
| Multiple linear regression (all features, training) | **0.658** |
| Scikit-learn polynomial pipeline (training) | **0.751** |
| Ridge regression, α=0.1 (test) | **0.648** |
| **2nd-order polynomial Ridge, α=0.1 (test)** | **0.700** |

- **`sqft_living` is the single strongest predictor** — alone it explains ~49% of price variance.
- Adding all features raises R² to ~0.66, and a **second-order polynomial Ridge model improves the held-out test R² to 0.70**, clearly outperforming the linear Ridge model (0.65).
- The **waterfront view** group contains the most extreme price outliers (visible in the boxplot).
- `sqft_above` shows a clear **positive linear relationship** with price (regplot).

### One limitation
- The data is a **single 12-month window (May 2014 – May 2015)** for **one county**. It does not capture market changes over time, interest-rate effects, or city-level heterogeneity, and therefore cannot be generalised to other housing markets or later years without re-fitting.

---

## 5. Instructions for Running the Analysis

### Prerequisites
- Python 3.9+
- `pip`

### Setup
```bash
# 1. Clone this repository
git clone https://github.com/DuanQingru/data-science-project.git
cd data-science-project

# 2. (Recommended) virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Getting the data
Download `kc_house_data.csv` from Kaggle (https://www.kaggle.com/harlfoxem/housesalesprediction) and place it in the `data/` folder. The CSV is **not** committed to this repo.

### Running the analysis
```bash
jupyter notebook
# Open and run: notebooks/kc_house_price_prediction.ipynb
```

### Expected outputs
- Two visualisations saved to `images/`:
  - `waterfront_boxplot.png` — price by waterfront view (outliers)
  - `sqft_above_regplot.png` — regression line of price vs. above-ground square footage
- Printed R² values for each model, matching the table above.

---

## Repository structure
```
data-science-project/
├── data/
│   └── kc_house_data.csv         # (download separately — not committed)
├── images/
│   ├── waterfront_boxplot.png
│   └── sqft_above_regplot.png
├── notebooks/
│   └── kc_house_price_prediction.ipynb
├── requirements.txt
└── README.md
```

## License
Code released under the MIT License. The Kaggle dataset remains the property of its original publisher and is used here for education.
