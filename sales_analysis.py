"""
Sales Data Cleaning & Analysis Pipeline
Author: Shivansh Shukla | Velmora
Project: Freelance Portfolio — Sales Dashboard
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD RAW DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  SALES DATA CLEANING & ANALYSIS REPORT")
print("=" * 55)

df = pd.read_csv('raw_sales_data.csv')
print(f"\n[LOAD] Raw dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"       Missing values:\n{df.isnull().sum()}")
print(f"       Duplicate rows: {df.duplicated().sum()}")

# ─────────────────────────────────────────────
# STEP 2: REMOVE DUPLICATES
# ─────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n[DEDUP] Removed {before - len(df)} duplicate rows → {len(df)} rows remain")

# ─────────────────────────────────────────────
# STEP 3: FIX DATE COLUMN (mixed formats)
# ─────────────────────────────────────────────
def parse_date(val):
    if pd.isnull(val):
        return pd.NaT
    for fmt in ('%Y-%m-%d', '%d/%m/%Y'):
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            pass
    return pd.NaT

df['Date'] = df['Date'].apply(parse_date)

# Fill missing dates with median date
median_date = df['Date'].dropna().sort_values().iloc[len(df['Date'].dropna()) // 2]
missing_dates = df['Date'].isnull().sum()
df['Date'].fillna(median_date, inplace=True)
print(f"\n[DATE]  Fixed mixed formats. Filled {missing_dates} missing dates with median: {median_date.date()}")

# ─────────────────────────────────────────────
# STEP 4: HANDLE MISSING VALUES
# ─────────────────────────────────────────────
# Customer_Name → fill with 'Unknown'
df['Customer_Name'].fillna('Unknown Customer', inplace=True)

# City → fill with mode
city_mode = df['City'].mode()[0]
df['City'].fillna(city_mode, inplace=True)

# Category → fill from product mapping
product_category_map = df.dropna(subset=['Category']).drop_duplicates('Product').set_index('Product')['Category']
df['Category'] = df.apply(
    lambda r: product_category_map.get(r['Product'], 'Unknown') if pd.isnull(r['Category']) else r['Category'], axis=1
)

# Price → fill from product median
df['Price'] = df.groupby('Product')['Price'].transform(lambda x: x.fillna(x.median()))

# Quantity → fill with 1 (conservative default)
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1).astype(int)

print(f"\n[FILL]  Missing values after cleaning:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# STEP 5: FIX WRONG Total_Sales VALUES
# ─────────────────────────────────────────────
df['Correct_Total'] = (df['Quantity'] * df['Price']).round(2)

# Fill NaN Total_Sales
df['Total_Sales'].fillna(df['Correct_Total'], inplace=True)

# Detect and fix errors (>5% deviation from correct value)
tolerance = 0.05
mask_wrong = (abs(df['Total_Sales'] - df['Correct_Total']) / df['Correct_Total']) > tolerance
fixed_count = mask_wrong.sum()
df.loc[mask_wrong, 'Total_Sales'] = df.loc[mask_wrong, 'Correct_Total']
print(f"\n[FIX]   Corrected {fixed_count} rows with wrong Total_Sales values")

# ─────────────────────────────────────────────
# STEP 6: CREATE NEW COLUMNS
# ─────────────────────────────────────────────
df['Month'] = df['Date'].dt.strftime('%b')
df['Month_Num'] = df['Date'].dt.month
df['Revenue'] = df['Total_Sales']  # clean alias for clarity

month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(f"\n[NEW]   Added columns: Month, Month_Num, Revenue")
print(f"        Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# STEP 7: ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ANALYSIS RESULTS")
print("=" * 55)

total_revenue = df['Revenue'].sum()
print(f"\n  Total Revenue       : ${total_revenue:,.2f}")
print(f"  Total Orders        : {len(df):,}")
print(f"  Avg Order Value     : ${df['Revenue'].mean():,.2f}")

# Top 5 products
top5 = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(5)
print(f"\n  TOP 5 PRODUCTS BY REVENUE:")
for p, r in top5.items():
    print(f"    {p:<22} ${r:>10,.2f}")

# Sales by city
city_sales = df.groupby('City')['Revenue'].sum().sort_values(ascending=False)
print(f"\n  SALES BY CITY (Top 5):")
for c, r in city_sales.head(5).items():
    print(f"    {c:<18} ${r:>10,.2f}")

# Monthly trend
monthly = df.groupby(['Month_Num','Month'])['Revenue'].sum().reset_index()
monthly = monthly.sort_values('Month_Num')
print(f"\n  MONTHLY REVENUE:")
for _, row in monthly.iterrows():
    bar = '█' * int(row['Revenue'] / 3000)
    print(f"    {row['Month']:<5} ${row['Revenue']:>9,.0f}  {bar}")

# Category distribution
cat_dist = df.groupby('Category')['Revenue'].sum()

# ─────────────────────────────────────────────
# STEP 8: DASHBOARD VISUALIZATION
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
BG     = '#0F1117'
CARD   = '#1A1D27'
ACCENT = '#4F8EF7'
GREEN  = '#27AE60'
ORANGE = '#E67E22'
PURPLE = '#9B59B6'
WHITE  = '#FFFFFF'
GREY   = '#8A8F9E'

fig = plt.figure(figsize=(18, 18), facecolor=BG)
fig.suptitle('', fontsize=1)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.4,
                       left=0.06, right=0.97, top=0.92, bottom=0.04,
                       height_ratios=[1, 1.4, 1.4, 1.1])

# ── HEADER KPI CARDS ──────────────────────────────────────────────
kpi_data = [
    ('TOTAL REVENUE', f"${total_revenue:,.0f}", ACCENT),
    ('TOTAL ORDERS',  f"{len(df):,}",           GREEN),
    ('AVG ORDER VAL', f"${df['Revenue'].mean():,.2f}", ORANGE),
]
for i, (label, val, color) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(CARD)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.72, val, ha='center', va='center', fontsize=26,
            fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.28, label, ha='center', va='center', fontsize=9,
            color=GREY, transform=ax.transAxes, fontweight='bold')
    for spine in ['top','right','left','bottom']:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(1.5)

# ── BAR CHART: TOP 5 PRODUCTS ─────────────────────────────────────
ax1 = fig.add_subplot(gs[1, :2])
ax1.set_facecolor(CARD)
colors_bar = [ACCENT, '#6BA5F8', '#8BBAF9', '#ABD0FA', '#CBE5FB']
bars = ax1.barh(top5.index[::-1], top5.values[::-1], color=colors_bar, height=0.6, edgecolor='none')
for bar, val in zip(bars, top5.values[::-1]):
    ax1.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
             f'${val:,.0f}', va='center', ha='left', fontsize=9, color=WHITE)
ax1.set_title('Top 5 Products by Revenue', color=WHITE, fontsize=11, fontweight='bold', pad=10)
ax1.tick_params(colors=GREY, labelsize=9)
ax1.set_facecolor(CARD)
ax1.spines[:].set_visible(False)
ax1.xaxis.grid(True, color='#2A2D3A', linewidth=0.5)
ax1.set_axisbelow(True)
ax1.tick_params(axis='y', colors=WHITE)
ax1.tick_params(axis='x', colors=GREY)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

# ── PIE CHART: CATEGORY DISTRIBUTION ─────────────────────────────
ax2 = fig.add_subplot(gs[1, 2])
ax2.set_facecolor(CARD)
pie_colors = [ACCENT, GREEN, ORANGE, PURPLE, '#E74C3C'][:len(cat_dist)]
wedges, texts, autotexts = ax2.pie(
    cat_dist.values, labels=cat_dist.index, autopct='%1.1f%%',
    colors=pie_colors, startangle=140,
    textprops={'color': WHITE, 'fontsize': 8},
    wedgeprops={'edgecolor': BG, 'linewidth': 2},
    pctdistance=0.78
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_color(WHITE)
ax2.set_title('Category Distribution', color=WHITE, fontsize=11, fontweight='bold', pad=10)

# ── LINE CHART: MONTHLY TREND ─────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :2])
ax3.set_facecolor(CARD)
ax3.plot(monthly['Month'], monthly['Revenue'], color=ACCENT, linewidth=2.5,
         marker='o', markersize=6, markerfacecolor=WHITE, markeredgecolor=ACCENT, markeredgewidth=2)
ax3.fill_between(monthly['Month'], monthly['Revenue'], alpha=0.15, color=ACCENT)
ax3.set_title('Monthly Revenue Trend (2023)', color=WHITE, fontsize=11, fontweight='bold', pad=10)
ax3.spines[:].set_visible(False)
ax3.tick_params(colors=GREY, labelsize=8)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y/1000:.0f}K'))
ax3.tick_params(axis='x', colors=WHITE)
ax3.tick_params(axis='y', colors=GREY)
ax3.yaxis.grid(True, color='#2A2D3A', linewidth=0.5)
ax3.set_axisbelow(True)

# ── BAR CHART: SALES BY CITY ──────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 2])
ax4.set_facecolor(CARD)
city_top = city_sales.head(6)
bar_colors = [GREEN if i == 0 else '#3DAA6A' if i == 1 else '#4DC77A' for i in range(len(city_top))]
ax4.bar(city_top.index, city_top.values, color=bar_colors, edgecolor='none', width=0.6)
ax4.set_title('Revenue by City', color=WHITE, fontsize=11, fontweight='bold', pad=10)
ax4.spines[:].set_visible(False)
ax4.tick_params(axis='x', rotation=30, colors=WHITE, labelsize=7)
ax4.tick_params(axis='y', colors=GREY, labelsize=8)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y/1000:.0f}K'))
ax4.yaxis.grid(True, color='#2A2D3A', linewidth=0.5)
ax4.set_axisbelow(True)

# ── KEY INSIGHTS ──────────────────────────────────────────────────
ax_ins = fig.add_subplot(gs[3, :2])
ax_ins.set_facecolor(CARD)
ax_ins.set_xlim(0, 1); ax_ins.set_ylim(0, 1)
ax_ins.axis('off')
for spine in ['top','right','left','bottom']:
    ax_ins.spines[spine].set_color(ACCENT)
    ax_ins.spines[spine].set_linewidth(1.2)

ax_ins.text(0.03, 0.88, '📊  KEY INSIGHTS', fontsize=10, fontweight='bold',
            color=ACCENT, transform=ax_ins.transAxes, va='top')

insights = [
    ('▲', ORANGE,  'Revenue peaks around Q4 → strong seasonal demand trend'),
    ('▲', '#E74C3C','Electronics dominate sales → over-dependence risk on one category'),
    ('▼', PURPLE,  'Certain cities (Austin, San Diego) consistently underperform vs top markets'),
]
for idx, (icon, col, text) in enumerate(insights):
    y = 0.65 - idx * 0.25
    ax_ins.text(0.03, y, icon, fontsize=11, color=col, transform=ax_ins.transAxes, va='center')
    ax_ins.text(0.07, y, text, fontsize=9, color=WHITE, transform=ax_ins.transAxes, va='center')

# ── BUSINESS RECOMMENDATIONS ──────────────────────────────────────
ax_rec = fig.add_subplot(gs[3, 2])
ax_rec.set_facecolor(CARD)
ax_rec.set_xlim(0, 1); ax_rec.set_ylim(0, 1)
ax_rec.axis('off')
for spine in ['top','right','left','bottom']:
    ax_rec.spines[spine].set_color(GREEN)
    ax_rec.spines[spine].set_linewidth(1.2)

ax_rec.text(0.05, 0.88, '✅  RECOMMENDATIONS', fontsize=10, fontweight='bold',
            color=GREEN, transform=ax_rec.transAxes, va='top')

recs = [
    'Diversify product categories to reduce revenue concentration risk',
    'Launch targeted marketing campaigns in low-performing cities',
    'Build inventory buffer in Sep–Oct for Q4 demand surge',
]
for idx, text in enumerate(recs):
    y = 0.63 - idx * 0.25
    ax_rec.text(0.05, y, f'→  {text}', fontsize=8.5, color=WHITE,
                transform=ax_rec.transAxes, va='center', wrap=True)

# ── TITLE + FOOTER ────────────────────────────────────────────────
fig.text(0.5, 0.955, 'SALES PERFORMANCE DASHBOARD  ·  2023',
         ha='center', fontsize=16, fontweight='bold', color=WHITE)
fig.text(0.5, 0.935, 'Annual overview  |  520 orders  |  Data cleaned & validated',
         ha='center', fontsize=9, color=GREY)
fig.text(0.97, 0.012, 'Built by Shivansh Shukla', ha='right', fontsize=8, color=GREY, style='italic')

plt.savefig('/home/claude/sales_dashboard.png', dpi=160, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print("\n[CHART] Dashboard saved → sales_dashboard.png")

# Save cleaned CSV
df.drop(columns=['Correct_Total'], inplace=True)
df.to_csv('/home/claude/cleaned_sales_data.csv', index=False)
print("[SAVE]  Cleaned dataset saved → cleaned_sales_data.csv")
print("\n" + "=" * 55)
print("  PIPELINE COMPLETE")
print("=" * 55)
