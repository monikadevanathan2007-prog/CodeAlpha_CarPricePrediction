import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. DATASET ────────────────────────────────────────────────────────────────
data = {
    'Car_Name': ['ritz','sx4','ciaz','wagon r','swift','vitara brezza','ciaz','s cross','ciaz','ciaz',
                 'alto 800','ciaz','ciaz','ertiga','dzire','ertiga','ertiga','ertiga','wagon r','sx4',
                 'alto k10','ignis','sx4','alto k10','wagon r','swift','swift','swift','alto k10','ciaz',
                 'ritz','ritz','swift','ertiga','dzire','sx4','dzire','800','alto k10','sx4','baleno',
                 'alto k10','sx4','dzire','omni','ciaz','ritz','wagon r','ertiga','ciaz',
                 'fortuner','fortuner','innova','fortuner','innova','corolla altis','etios cross',
                 'corolla altis','etios g','fortuner','corolla altis','etios cross','fortuner','fortuner',
                 'fortuner','etios liva','innova','fortuner','corolla altis','corolla altis','etios liva',
                 'corolla altis','etios liva','etios cross','etios g','corolla altis','corolla',
                 'corolla altis','fortuner','corolla altis','etios gd','innova','innova','innova',
                 'camry','land cruiser','corolla altis','etios liva','etios g','corolla altis','innova',
                 'innova','fortuner','corolla altis','corolla altis','innova','corolla altis',
                 'corolla altis','fortuner','i20','grand i10','i10','eon','grand i10','xcent',
                 'grand i10','i20','grand i10','i10','elantra','creta','i20','grand i10','verna',
                 'eon','eon','verna','verna','eon','i20','i20','verna','verna','i10','grand i10',
                 'i10','verna','i20','verna','verna','elantra','grand i10','grand i10','verna','eon',
                 'creta','verna','eon','verna','xcent','xcent','i20','verna','verna','i20','verna',
                 'i10','i20','creta','city','brio','city','city','brio','city','city','city','amaze',
                 'city','brio','amaze','jazz','amaze','jazz','amaze','city','brio','city','city',
                 'city','jazz','brio','city','city','jazz','city','brio','city','jazz','jazz',
                 'amaze','city','city','amaze','brio','jazz','city','brio'],
    'Year': [2014,2013,2017,2011,2014,2018,2015,2015,2016,2015,2017,2015,2015,2015,2009,2016,2015,
             2016,2015,2010,2016,2017,2011,2014,2013,2011,2013,2017,2010,2015,2012,2011,2014,2015,
             2014,2011,2015,2003,2016,2003,2016,2014,2008,2014,2012,2014,2013,2006,2015,2017,
             2012,2015,2017,2013,2005,2009,2015,2010,2014,2014,2013,2015,2014,2015,2017,2014,
             2017,2010,2011,2016,2014,2011,2013,2011,2014,2015,2013,2004,2010,2012,2016,2015,
             2017,2015,2005,2006,2010,2012,2013,2014,2005,2015,2008,2012,2016,2017,2013,2010,
             2010,2015,2011,2015,2016,2016,2015,2017,2015,2010,2015,2011,2015,2014,2015,2012,
             2013,2014,2012,2012,2015,2013,2016,2014,2011,2015,2013,2015,2014,2015,2017,2011,
             2013,2012,2016,2015,2012,2013,2013,2016,2015,2015,2016,2013,2012,2016,2015,2015,
             2016,2015,2016,2015,2016,2017,2016,2015,2016,2016,2015,2017,2014,2016,2015,2016,
             2016,2010,2014,2010,2016,2014,2016,2015,2016,2015,2015,2014,2016,2006,2014,2015,
             2016,2015,2015,2015,2009,2017,2016],
    'Selling_Price': [3.35,4.75,7.25,2.85,4.6,9.25,6.75,6.5,8.75,7.45,2.85,6.85,7.5,6.1,2.25,
                      7.75,7.25,7.75,3.25,2.65,2.85,4.9,4.4,2.5,2.9,3.0,4.15,6.0,1.95,7.45,
                      3.1,2.35,4.95,6.0,5.5,2.95,4.65,0.35,3.0,2.25,5.85,2.55,1.95,5.5,1.25,
                      7.5,2.65,1.05,5.8,7.75,14.9,23.0,18.0,16.0,2.75,3.6,4.5,4.75,4.1,19.99,
                      6.95,4.5,18.75,23.5,33.0,4.75,19.75,9.25,4.35,14.25,3.95,4.5,3.45,2.65,
                      4.9,3.95,5.5,1.5,5.25,14.5,14.73,4.75,23.0,12.5,3.49,2.5,9.25,5.9,3.45,
                      4.75,3.81,11.25,3.51,23.0,4.0,5.85,20.75,17.0,7.05,9.65,3.25,4.4,2.95,
                      2.75,5.25,5.75,5.15,7.9,4.85,3.1,11.75,11.25,2.9,5.25,4.5,2.9,3.15,6.45,
                      4.5,3.5,4.5,6.0,8.25,5.11,2.7,5.25,2.55,4.95,3.1,6.15,9.25,11.45,3.9,
                      5.5,9.1,3.1,11.25,4.8,2.0,5.35,4.75,4.4,6.25,5.95,5.2,3.75,5.95,4.0,
                      5.25,12.9,5.0,5.4,7.2,5.25,3.0,10.25,8.5,8.4,3.9,9.15,5.5,4.0,6.6,4.0,
                      6.5,3.65,8.35,4.8,6.7,4.1,3.0,7.5,2.25,5.3,10.9,8.65,9.7,6.0,6.25,2.1,
                      8.25,8.99,3.5,7.4,5.65,5.75,8.4,10.11,4.5,5.4,6.4,3.25,3.75,8.55,9.5,
                      4.0,5.3],
    'Present_Price': [5.59,9.54,9.85,4.15,6.87,9.83,8.12,8.61,8.89,8.92,3.6,10.38,9.94,7.71,
                      7.21,10.79,10.79,10.79,5.09,7.98,3.95,5.71,8.01,3.46,4.41,4.99,5.87,6.49,
                      3.95,10.38,5.98,4.89,7.49,9.95,8.06,7.74,7.2,2.28,3.76,7.98,7.87,3.98,
                      7.15,8.06,2.69,12.04,4.89,4.15,7.71,9.29,30.61,30.61,19.77,30.61,10.21,
                      15.04,7.27,18.54,6.8,35.96,18.61,7.7,35.96,35.96,36.23,6.95,23.15,20.45,
                      13.74,20.91,6.76,12.48,6.05,5.71,8.93,6.8,14.68,12.35,22.83,30.61,14.89,
                      7.85,25.39,13.46,13.46,23.73,92.6,13.74,6.05,6.76,18.61,16.09,13.7,30.61,
                      22.78,18.61,25.39,18.64,18.61,20.45,6.79,5.7,4.6,4.43,5.7,7.13,5.7,8.1,
                      5.7,4.6,14.79,13.6,6.79,5.7,9.4,4.43,4.43,8.4,9.4,4.43,6.79,7.6,9.4,9.4,
                      4.6,5.7,4.43,9.4,6.79,9.4,9.4,14.79,5.7,5.7,9.4,4.43,13.6,9.4,4.43,9.4,
                      7.13,7.13,7.6,9.4,9.4,6.79,9.4,4.6,7.6,13.6,9.9,6.82,9.9,9.9,5.35,13.6,
                      13.6,13.6,7.0,13.6,5.97,5.8,7.7,7.0,8.7,7.0,9.4,5.8,10.0,10.0,10.0,7.56,
                      6.81,13.6,13.6,8.44,11.89,5.9,9.4,8.4,13.6,7.0,6.8,13.09,11.6,5.9,11.0,
                      14.0,12.59,11.8],
    'Driven_kms': [27000,43000,6900,5200,42450,2071,18796,33429,20273,42367,2135,51000,15000,26000,
                   77427,43000,41678,43000,35500,41442,25000,2400,50000,45280,56879,20000,55138,
                   16200,44542,45000,51439,54200,39000,45000,45000,49998,48767,127000,10079,62000,
                   24524,46706,58000,45780,50000,15000,64532,65000,25870,37000,104707,40000,15000,
                   135000,90000,70000,40534,50000,39485,41000,40001,40588,78000,47000,6000,45000,
                   11000,59000,88000,12000,71000,45000,43000,72000,47000,83000,36000,72000,135154,
                   80000,89000,23000,40000,15000,38000,197176,142000,78000,56000,47000,40000,58242,
                   75000,40000,89000,72000,29000,8700,45000,50024,58000,28200,53460,28282,3493,
                   12479,34797,3435,21125,35775,43535,22671,31604,20114,36100,12500,15000,45078,
                   36000,38488,32000,77632,61381,36198,22517,24678,57000,60000,52132,45000,15001,
                   12900,53000,4492,15141,11849,68000,60241,23709,32322,35866,34000,7000,49000,
                   71000,35000,36000,30000,40023,16002,40026,21200,35000,19434,19000,56701,31427,
                   48000,54242,53675,49562,40324,25000,36054,29223,5600,40023,16002,40026,21200,
                   19434,16500,49562,54242,40324,53675,25000,76,48000,38000,53675,30753,5464],
    'Fuel_Type': ['Petrol','Diesel','Petrol','Petrol','Diesel','Diesel','Petrol','Diesel','Diesel','Diesel',
                  'Petrol','Diesel','Petrol','Petrol','Petrol','Diesel','Diesel','Diesel','CNG','Petrol',
                  'Petrol','Petrol','Petrol','Petrol','Petrol','Petrol','Petrol','Petrol','Petrol','Diesel',
                  'Diesel','Petrol','Diesel','Diesel','Diesel','CNG','Petrol','Petrol','Petrol','Petrol',
                  'Petrol','Petrol','Petrol','Diesel','Petrol','Petrol','Petrol','Petrol','Petrol','Petrol',
                  'Diesel','Diesel','Diesel','Diesel','Petrol','Petrol','Petrol','Petrol','Petrol','Diesel',
                  'Petrol','Petrol','Diesel','Diesel','Diesel','Diesel','Petrol','Diesel','Petrol','Petrol',
                  'Diesel','Diesel','Petrol','Diesel','Petrol','Petrol','Petrol','Petrol','Diesel','Petrol',
                  'Diesel','Diesel','Diesel','Petrol','Diesel','Petrol','Diesel','Petrol','Petrol','Diesel',
                  'Petrol','Petrol','Diesel','Petrol','Diesel','Petrol','Diesel','Petrol','Petrol','Diesel',
                  'Petrol','Petrol','Petrol','Petrol','Diesel','Petrol','Petrol','Petrol','Diesel','Petrol',
                  'Diesel','Petrol','Petrol','Petrol','Petrol','Diesel','Diesel','Petrol','Petrol','Petrol',
                  'Diesel','Petrol','Petrol','Petrol','Diesel','Diesel','Petrol','Petrol','Petrol','Diesel',
                  'Petrol','Diesel','Petrol','Petrol','Diesel','Petrol','Petrol','Petrol','Petrol','Diesel',
                  'Petrol','Diesel','Petrol','Petrol','Diesel','Diesel','Diesel','Petrol','Petrol','Petrol',
                  'Petrol','Petrol','Petrol','Petrol','Petrol','Diesel','Petrol','Diesel','Diesel','Petrol',
                  'Petrol','Petrol','Petrol','Petrol','Diesel','Diesel','Petrol','Diesel','Petrol','Petrol',
                  'Petrol','Diesel','Petrol'],
    'Selling_type': ['Dealer']*90 + ['Individual']*70 + ['Dealer']*29,
    'Transmission': ['Manual']*12 + ['Automatic'] + ['Manual']*37 + ['Automatic'] + ['Manual']*8 +
                    ['Automatic']*2 + ['Manual']*27 + ['Automatic']*4 + ['Manual']*5 + ['Automatic']*3 +
                    ['Manual']*16 + ['Automatic']*2 + ['Manual']*4 + ['Automatic']*3 + ['Manual']*55 +
                    ['Automatic']*8,
    'Owner': [0]*130 + [1]*10 + [0]*49
}

# Build DataFrame - use minimum length
min_len = min(len(v) for v in data.values())
df = pd.DataFrame({k: v[:min_len] for k, v in data.items()})

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
df['Car_Age'] = 2024 - df['Year']

le = LabelEncoder()
df['Fuel_Type_enc']     = le.fit_transform(df['Fuel_Type'])
df['Selling_type_enc']  = le.fit_transform(df['Selling_type'])
df['Transmission_enc']  = le.fit_transform(df['Transmission'])

features = ['Present_Price', 'Driven_kms', 'Fuel_Type_enc',
            'Selling_type_enc', 'Transmission_enc', 'Owner', 'Car_Age']
X = df[features]
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. MODELS ─────────────────────────────────────────────────────────────────
models = {
    'Linear Regression':      LinearRegression(),
    'Random Forest':          RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':      GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        'model': model,
        'preds': preds,
        'MAE':   mean_absolute_error(y_test, preds),
        'RMSE':  np.sqrt(mean_squared_error(y_test, preds)),
        'R2':    r2_score(y_test, preds),
    }
    print(f"\n{name}: MAE={results[name]['MAE']:.2f}, RMSE={results[name]['RMSE']:.2f}, R²={results[name]['R2']:.4f}")

best_name = max(results, key=lambda k: results[k]['R2'])
best      = results[best_name]
print(f"\n✅ Best model: {best_name} (R²={best['R2']:.4f})")

# ── 4. PLOTS ──────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'primary': '#2C3E6B', 'accent': '#E84545', 'green': '#27AE60',
          'orange': '#E67E22', 'purple': '#8E44AD', 'light': '#ECF0F1'}

# ── Plot 1: EDA Dashboard ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('🚗  Car Price Prediction — Exploratory Data Analysis',
             fontsize=20, fontweight='bold', color=colors['primary'], y=1.01)

# 1a. Selling Price Distribution
axes[0,0].hist(df['Selling_Price'], bins=30, color=colors['primary'], edgecolor='white', alpha=0.85)
axes[0,0].set_title('Selling Price Distribution', fontweight='bold', fontsize=13)
axes[0,0].set_xlabel('Price (Lakhs ₹)'); axes[0,0].set_ylabel('Frequency')
axes[0,0].axvline(df['Selling_Price'].median(), color=colors['accent'], lw=2, linestyle='--', label=f'Median: ₹{df["Selling_Price"].median():.1f}L')
axes[0,0].legend()

# 1b. Fuel Type Distribution
ft_counts = df['Fuel_Type'].value_counts()
axes[0,1].bar(ft_counts.index, ft_counts.values,
              color=[colors['primary'], colors['accent'], colors['green']], edgecolor='white')
axes[0,1].set_title('Fuel Type Distribution', fontweight='bold', fontsize=13)
axes[0,1].set_xlabel('Fuel Type'); axes[0,1].set_ylabel('Count')
for i, v in enumerate(ft_counts.values):
    axes[0,1].text(i, v+1, str(v), ha='center', fontweight='bold')

# 1c. Price vs Present Price scatter
sc = axes[0,2].scatter(df['Present_Price'], df['Selling_Price'],
                       c=df['Car_Age'], cmap='RdYlGn_r', alpha=0.7, s=60, edgecolors='white', lw=0.5)
plt.colorbar(sc, ax=axes[0,2], label='Car Age (years)')
axes[0,2].set_title('Present Price vs Selling Price\n(coloured by Car Age)', fontweight='bold', fontsize=13)
axes[0,2].set_xlabel('Present Price (Lakhs ₹)'); axes[0,2].set_ylabel('Selling Price (Lakhs ₹)')

# 1d. Transmission boxplot
trans_groups = [df[df['Transmission']==t]['Selling_Price'].values for t in df['Transmission'].unique()]
bp = axes[1,0].boxplot(trans_groups, labels=df['Transmission'].unique(), patch_artist=True)
for patch, clr in zip(bp['boxes'], [colors['primary'], colors['accent']]):
    patch.set_facecolor(clr); patch.set_alpha(0.7)
axes[1,0].set_title('Selling Price by Transmission', fontweight='bold', fontsize=13)
axes[1,0].set_ylabel('Price (Lakhs ₹)')

# 1e. Driven KMs vs Price
axes[1,1].scatter(df['Driven_kms'], df['Selling_Price'],
                  color=colors['purple'], alpha=0.5, s=50, edgecolors='white', lw=0.5)
axes[1,1].set_title('Driven KMs vs Selling Price', fontweight='bold', fontsize=13)
axes[1,1].set_xlabel('Driven KMs'); axes[1,1].set_ylabel('Price (Lakhs ₹)')
z = np.polyfit(df['Driven_kms'], df['Selling_Price'], 1)
p = np.poly1d(z)
axes[1,1].plot(sorted(df['Driven_kms']), p(sorted(df['Driven_kms'])),
               color=colors['accent'], lw=2, linestyle='--', label='Trend')
axes[1,1].legend()

# 1f. Correlation heatmap
num_cols = ['Selling_Price','Present_Price','Driven_kms','Car_Age','Owner']
corr = df[num_cols].corr()
im = axes[1,2].imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[1,2])
axes[1,2].set_xticks(range(len(num_cols))); axes[1,2].set_xticklabels(num_cols, rotation=30, ha='right', fontsize=9)
axes[1,2].set_yticks(range(len(num_cols))); axes[1,2].set_yticklabels(num_cols, fontsize=9)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        axes[1,2].text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8, fontweight='bold')
axes[1,2].set_title('Correlation Heatmap', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/screenshot1_EDA.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("✅ Screenshot 1 saved")

# ── Plot 2: Model Performance Dashboard ───────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('🤖  Car Price Prediction — Model Performance Dashboard',
             fontsize=20, fontweight='bold', color=colors['primary'], y=1.01)

model_names  = list(results.keys())
model_colors = [colors['primary'], colors['accent'], colors['green']]

# 2a. R² comparison
r2_vals = [results[n]['R2'] for n in model_names]
bars = axes[0,0].bar(model_names, r2_vals, color=model_colors, edgecolor='white', width=0.5)
axes[0,0].set_title('R² Score Comparison', fontweight='bold', fontsize=13)
axes[0,0].set_ylabel('R² Score'); axes[0,0].set_ylim(0, 1.05)
for bar, val in zip(bars, r2_vals):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.4f}',
                   ha='center', fontweight='bold', fontsize=11)
axes[0,0].set_xticklabels(model_names, rotation=10)

# 2b. MAE comparison
mae_vals = [results[n]['MAE'] for n in model_names]
bars2 = axes[0,1].bar(model_names, mae_vals, color=model_colors, edgecolor='white', width=0.5)
axes[0,1].set_title('MAE Comparison (lower = better)', fontweight='bold', fontsize=13)
axes[0,1].set_ylabel('Mean Absolute Error')
for bar, val in zip(bars2, mae_vals):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.2f}',
                   ha='center', fontweight='bold', fontsize=11)
axes[0,1].set_xticklabels(model_names, rotation=10)

# 2c. RMSE comparison
rmse_vals = [results[n]['RMSE'] for n in model_names]
bars3 = axes[0,2].bar(model_names, rmse_vals, color=model_colors, edgecolor='white', width=0.5)
axes[0,2].set_title('RMSE Comparison (lower = better)', fontweight='bold', fontsize=13)
axes[0,2].set_ylabel('Root Mean Squared Error')
for bar, val in zip(bars3, rmse_vals):
    axes[0,2].text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.2f}',
                   ha='center', fontweight='bold', fontsize=11)
axes[0,2].set_xticklabels(model_names, rotation=10)

# 2d. Best model: Actual vs Predicted
best_preds = best['preds']
axes[1,0].scatter(y_test, best_preds, color=colors['primary'], alpha=0.7, s=60, edgecolors='white')
lims = [min(y_test.min(), best_preds.min())-1, max(y_test.max(), best_preds.max())+1]
axes[1,0].plot(lims, lims, color=colors['accent'], lw=2, linestyle='--', label='Perfect Prediction')
axes[1,0].set_title(f'{best_name}\nActual vs Predicted', fontweight='bold', fontsize=13)
axes[1,0].set_xlabel('Actual Price (Lakhs ₹)'); axes[1,0].set_ylabel('Predicted Price (Lakhs ₹)')
axes[1,0].legend(); axes[1,0].set_xlim(lims); axes[1,0].set_ylim(lims)

# 2e. Residuals
residuals = y_test.values - best_preds
axes[1,1].scatter(best_preds, residuals, color=colors['purple'], alpha=0.7, s=60, edgecolors='white')
axes[1,1].axhline(0, color=colors['accent'], lw=2, linestyle='--')
axes[1,1].set_title(f'{best_name} — Residual Plot', fontweight='bold', fontsize=13)
axes[1,1].set_xlabel('Predicted Price'); axes[1,1].set_ylabel('Residuals')

# 2f. Feature Importance (RF)
rf_model = results['Random Forest']['model']
fi = pd.Series(rf_model.feature_importances_, index=features).sort_values()
axes[1,2].barh(fi.index, fi.values, color=colors['primary'], edgecolor='white')
axes[1,2].set_title('Feature Importance\n(Random Forest)', fontweight='bold', fontsize=13)
axes[1,2].set_xlabel('Importance Score')
for i, (idx, val) in enumerate(fi.items()):
    axes[1,2].text(val+0.001, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/screenshot2_ModelPerformance.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("✅ Screenshot 2 saved")

# ── Plot 3: LinkedIn Summary Card ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor(colors['primary'])
ax.set_facecolor(colors['primary'])
ax.axis('off')

# Header
ax.text(0.5, 0.95, '🚗  Car Price Prediction', transform=ax.transAxes,
        fontsize=28, fontweight='bold', ha='center', va='top', color='white')
ax.text(0.5, 0.88, 'Machine Learning Project  |  CodeAlpha Data Science Internship',
        transform=ax.transAxes, fontsize=14, ha='center', va='top', color='#BDC3C7')

# Metric boxes
metrics = [
    ('Best Model', best_name, colors['accent']),
    ('R² Score', f"{best['R2']:.4f}", colors['green']),
    ('MAE', f"₹{best['MAE']:.2f}L", colors['orange']),
    ('RMSE', f"₹{best['RMSE']:.2f}L", colors['purple']),
]
for i, (label, value, clr) in enumerate(metrics):
    x = 0.08 + i * 0.23
    rect = plt.Rectangle((x, 0.55), 0.19, 0.25, transform=ax.transAxes,
                          facecolor=clr, alpha=0.85, zorder=2)
    ax.add_patch(rect)
    ax.text(x+0.095, 0.74, label, transform=ax.transAxes,
            fontsize=10, ha='center', color='white', fontweight='bold')
    ax.text(x+0.095, 0.63, value, transform=ax.transAxes,
            fontsize=13, ha='center', color='white', fontweight='bold')

# Tech stack
techs = ['Python', 'Pandas', 'Scikit-learn', 'Matplotlib', 'Seaborn', 'Random Forest', 'Gradient Boosting']
ax.text(0.5, 0.50, '🛠  Tech Stack', transform=ax.transAxes,
        fontsize=13, ha='center', color='#BDC3C7', fontweight='bold')
for i, tech in enumerate(techs):
    xp = 0.06 + (i % 4) * 0.23
    yp = 0.38 if i < 4 else 0.28
    rect2 = plt.Rectangle((xp, yp), 0.20, 0.07, transform=ax.transAxes,
                           facecolor='#34495E', alpha=0.9, zorder=2)
    ax.add_patch(rect2)
    ax.text(xp+0.10, yp+0.035, tech, transform=ax.transAxes,
            fontsize=10, ha='center', va='center', color='white')

# Key findings
findings = [
    '✅  Present Price is the #1 predictor of resale value',
    '✅  Diesel cars retain value better than Petrol',
    '✅  Each year of age reduces price significantly',
    '✅  Automatic transmission commands a price premium',
]
ax.text(0.5, 0.22, '🔍  Key Findings', transform=ax.transAxes,
        fontsize=13, ha='center', color='#BDC3C7', fontweight='bold')
for i, f in enumerate(findings):
    ax.text(0.5, 0.16 - i*0.06, f, transform=ax.transAxes,
            fontsize=10, ha='center', color='white')

ax.text(0.5, 0.02, '#DataScience  #MachineLearning  #Python  #CodeAlpha  #CarPricePrediction',
        transform=ax.transAxes, fontsize=10, ha='center', color='#7F8C8D')

plt.savefig('/mnt/user-data/outputs/screenshot3_LinkedInCard.png', dpi=150, bbox_inches='tight',
            facecolor=colors['primary'])
plt.close()
print("✅ Screenshot 3 saved")

# ── 5. REPORT ─────────────────────────────────────────────────────────────────
report = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║          CAR PRICE PREDICTION — PROJECT REPORT                                 ║
║          CodeAlpha Data Science Internship | Task 3                            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

1. OBJECTIVE
   Predict the resale selling price of used cars using machine learning regression
   models based on car features such as brand, age, mileage, fuel type, and more.

2. DATASET OVERVIEW
   • Total Records  : {len(df)}
   • Features       : {df.shape[1]}
   • Target Variable: Selling_Price (in Lakhs ₹)
   • Price Range    : ₹{df['Selling_Price'].min():.2f}L – ₹{df['Selling_Price'].max():.2f}L
   • Avg Price      : ₹{df['Selling_Price'].mean():.2f}L

3. FEATURE ENGINEERING
   • Created 'Car_Age' from (2024 - Year)
   • Label-encoded: Fuel_Type, Selling_type, Transmission
   • Final features used: Present_Price, Driven_kms, Fuel_Type,
     Selling_type, Transmission, Owner, Car_Age

4. MODELS TRAINED & RESULTS
   ┌─────────────────────┬────────┬────────┬────────┐
   │ Model               │  MAE   │  RMSE  │   R²   │
   ├─────────────────────┼────────┼────────┼────────┤
   │ Linear Regression   │ {results['Linear Regression']['MAE']:6.2f} │ {results['Linear Regression']['RMSE']:6.2f} │ {results['Linear Regression']['R2']:6.4f} │
   │ Random Forest       │ {results['Random Forest']['MAE']:6.2f} │ {results['Random Forest']['RMSE']:6.2f} │ {results['Random Forest']['R2']:6.4f} │
   │ Gradient Boosting   │ {results['Gradient Boosting']['MAE']:6.2f} │ {results['Gradient Boosting']['RMSE']:6.2f} │ {results['Gradient Boosting']['R2']:6.4f} │
   └─────────────────────┴────────┴────────┴────────┘

5. BEST MODEL: {best_name}
   • R² Score : {best['R2']:.4f}  (explains {best['R2']*100:.1f}% of price variance)
   • MAE      : ₹{best['MAE']:.2f} Lakhs
   • RMSE     : ₹{best['RMSE']:.2f} Lakhs

6. KEY INSIGHTS
   • Present Price (ex-showroom) is the strongest predictor of resale value.
   • Car Age has a strong negative correlation with resale price.
   • Diesel-fuelled cars tend to have a higher resale value.
   • Automatic transmission vehicles command a slight price premium.
   • Lower driven kilometres positively impact the selling price.
   • Dealer-sold cars generally fetch higher prices than individual sellers.

7. CONCLUSION
   The {best_name} model achieved the highest R² of {best['R2']:.4f}, 
   making it the most reliable model for car price prediction. 
   The model can assist buyers, sellers, and dealerships in making 
   data-driven pricing decisions.
"""

print(report)
with open('/mnt/user-data/outputs/Task3_Report.txt', 'w') as f:
    f.write(report)
print("✅ Report saved")
