import geopandas as gpd
import pandas as pd
import numpy as np
from fredapi import Fred
from census import Census
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from tqdm import tqdm

# Enable tqdm for pandas
tqdm.pandas()

fred = Fred(api_key='123')
c = Census('2139f822b')

# load income data from census api
# Try ACS 1-year estimates (2005+, annual but less reliable) then fall back to 5-year (2009+, more reliable)
years = list(range(2005, 2025))
income_data = []

for y in tqdm(years, desc="Fetching Census income data"):
    try:
        # Try 1-year estimate first (available 2005+)
        tracts_income = c.acs1.state_county_tract(
            'B19013_001E',  # median household income
            state_fips='53',  # Washington state
            county_fips='*',  # all counties
            tract='*',
            year=y
        )
        source = 'acs1'
    except:
        try:
            # Fall back to 5-year estimate (available 2009+, more reliable)
            tracts_income = c.acs5.state_county_tract(
                'B19013_001E',
                state_fips='53',
                county_fips='*',
                tract='*',
                year=y
            )
            source = 'acs5'
        except:
            continue
    
    for t in tracts_income:
        income_data.append({
            'geoid': t['state'] + t['county'] + t['tract'],
            'year': y,
            'median_household_income': t['B19013_001E'],
            'source': source
        })

df_income = pd.DataFrame(income_data)
df_income['year'] = df_income['year'].astype(int)
df_income['geoid'] = df_income['geoid'].astype(str)

# Clean implausible ACS values (ACS sometimes encodes missing with large negative numbers)
# Keep only incomes in a reasonable band (0, 1e7]; drop the rest
valid_mask = (df_income['median_household_income'] > 0) & (df_income['median_household_income'] <= 1e7)
invalid_count = (~valid_mask).sum()
if invalid_count:
    print(f"Dropping {invalid_count} tract-year rows with invalid income values")
print(f"Income data coverage: {df_income['year'].min():.0f} - {df_income['year'].max():.0f}")
print(f"  ACS 1-year: {len(df_income[df_income['source']=='acs1'])} rows")
print(f"  ACS 5-year: {len(df_income[df_income['source']=='acs5'])} rows\n")
df_income = df_income[valid_mask].copy()

# CPI-U for inflation adjustment
cpi = fred.get_series('CPIAUCSL')
cpi = cpi.resample('YE').last().to_frame(name='cpi_index')
cpi['year'] = cpi.index.year
cpi = cpi.reset_index(drop=True)

# Base year CPI (e.g., 2023)
base_cpi = cpi.loc[cpi['year']==2023, 'cpi_index'].values[0]

# Merge CPI with income
df_income = df_income.merge(cpi[['year','cpi_index']], on='year', how='left')
df_income['median_household_income_adjusted'] = df_income['median_household_income'] * (base_cpi / df_income['cpi_index'])

# Keep only needed columns
df_income = df_income[['geoid','year','median_household_income_adjusted']]

# load tracts.shp 
tracts = gpd.read_file('tl_2020_53_tract.shp')

# Load the shapefile
park_sa_over_years = gpd.read_file('parks_sa_over_years.shp')

# Reproject to metric CRS for area calculations
tracts = tracts.to_crs(epsg=3857)
park_sa_over_years = park_sa_over_years.to_crs(epsg=3857)

# Prepare tracts geodataframe
tracts['geoid'] = tracts['GEOID']
tracts['area_m2'] = tracts.geometry.area
tracts['state_fips'] = tracts['STATEFP']
tracts = tracts[['geoid', 'geometry', 'area_m2', 'state_fips']]

# =========================================
# SIMPLE DiD: Compare income changes pre/post
# for tracts that got new parks vs those that didn't
# =========================================

print("\nIdentifying treatment status...")

# Use ALL parks (from 2003-2025) to identify treatment
# A tract is treated if it has >=0.01% overlap with parks established by 2025
parks_all_years = park_sa_over_years[(park_sa_over_years['est_year'].notna()) & 
                                      (park_sa_over_years['est_year'] >= 2003) &
                                      (park_sa_over_years['est_year'] <= 2025)]

print(f"Total parks in dataset (2003-2025): {len(parks_all_years)}")
print(f"Park establishment year range: {parks_all_years['est_year'].min():.0f} - {parks_all_years['est_year'].max():.0f}")

# For each tract, find first year (2003+) where cumulative overlap >= 0.01% of tract area
tracts['treated'] = 0
tracts['treat_year'] = pd.NA
tracts['treat_overlap_pct'] = np.nan

years_sorted = sorted(parks_all_years['est_year'].unique()) if not parks_all_years.empty else []
print(f"Park establishment years in dataset: {years_sorted[:20]}..." if len(years_sorted) > 20 else f"Park establishment years: {years_sorted}")

cumulative_union = None

for y in years_sorted:
    parks_up_to_y = parks_all_years[parks_all_years['est_year'] <= y]
    if parks_up_to_y.empty:
        continue
    try:
        # Use union_all() instead of deprecated unary_union
        cumulative_union = parks_up_to_y.geometry.union_all()
    except Exception as e:
        print(f"  Warning: topology error in year {y}, skipping: {e}")
        continue

    remaining_idx = tracts.index[tracts['treated'] == 0]
    if len(remaining_idx) == 0:
        break

    def overlap_pct(geom):
        inter_area = geom.intersection(cumulative_union).area
        return inter_area / geom.area if geom.area > 0 else 0.0

    overlaps = tracts.loc[remaining_idx, 'geometry'].apply(overlap_pct)
    newly_treated = overlaps[overlaps >= 0.0001]

    if not newly_treated.empty:
        tracts.loc[newly_treated.index, 'treated'] = 1
        tracts.loc[newly_treated.index, 'treat_year'] = y
        tracts.loc[newly_treated.index, 'treat_overlap_pct'] = newly_treated.values * 100
        print(f"  Year {y}: {len(newly_treated)} newly treated tracts")

print(f"\nTreated tracts (overlap >=0.01% by 2024): {tracts['treated'].sum()}")
print(f"Control tracts (no overlap >=0.01%): {(1-tracts['treated']).sum()}")

# =========================================
# Event-study: Identify top 3 treatment cohorts
# =========================================
print("\n" + "="*70)
print("IDENTIFYING TOP 3 TREATMENT COHORTS BY SIZE")
print("="*70)

# Count tracts treated per year
treat_counts = tracts[tracts['treated'] == 1]['treat_year'].value_counts().sort_values(ascending=False)
print(f"\nTreatment cohort sizes by year:\n{treat_counts}\n")

# Check available income years in data
available_years = sorted(df_income['year'].unique())
print(f"Available income data years: {available_years[0]:.0f} - {available_years[-1]:.0f}\n")

# Filter to only treatment years with sufficient data:
# - Must have pre-period data (treatment year >= min available year)
# - Must have post-period data (treatment year < max year - 3)
min_year = available_years[0]
max_year = available_years[-1] - 3

valid_treat_years = [y for y in treat_counts.index if y >= min_year and y <= max_year]
print(f"Valid treatment years (with pre/post data available): {sorted(valid_treat_years, reverse=True)}\n")

# From valid years, take the top 3 by cohort size
if valid_treat_years:
    valid_cohort_sizes = treat_counts[treat_counts.index.isin(valid_treat_years)].sort_values(ascending=False)
    top_years = valid_cohort_sizes.head(3).index.tolist()
    print(f"Top 3 valid treatment years (by cohort size):\n{valid_cohort_sizes.head(3)}\n")
else:
    print(f"Warning: No valid treatment years between {min_year} and {max_year}.")
    top_years = []

# =========================================
# Build DiD dataset for each cohort
# =========================================
rel_years = list(range(-3, 4))  # -3, -2, -1, 0, 1, 2, 3

all_event_study_results = []

for rank, treat_year in enumerate(top_years, 1):
    print(f"\n{'='*70}")
    print(f"COHORT {rank}: Treatment Year = {treat_year}")
    print(f"{'='*70}")
    
    # Get treated tracts for this cohort
    treated_geoids = tracts[tracts['treat_year'] == treat_year]['geoid'].unique()
    n_treated = len(treated_geoids)
    
    print(f"Number of treated tracts: {n_treated}")
    
    if n_treated == 0:
        print(f"No tracts for year {treat_year}, skipping.")
        continue
    
    # Get all treated tracts' income at years -3, -2, -1, 0 (relative to treatment year)
    # But only use years that exist in the income data
    ref_years_all = [treat_year - 3, treat_year - 2, treat_year - 1, treat_year]
    ref_years_all = [y for y in ref_years_all if y in available_years]
    
    if not ref_years_all:
        print(f"No reference years available near treatment year {treat_year}, skipping.")
        continue
    
    treated_ref_all = df_income[(df_income['geoid'].isin(treated_geoids)) & 
                                (df_income['year'].isin(ref_years_all))][['geoid', 'median_household_income_adjusted']]
    
    if treated_ref_all.empty:
        print(f"No income data for treated tracts in years {ref_years_all}, skipping.")
        continue
    
    # Compute median income across the entire pre/at-treatment window
    treated_avg_income = treated_ref_all.groupby('geoid')['median_household_income_adjusted'].mean()
    treat_median = treated_avg_income.median()
    treat_mean = treated_avg_income.mean()
    
    print(f"Treated cohort avg income (years {ref_years_all}): ${treat_median:,.0f} (median)")
    
    # Find control tracts: not treated, collect those across all ref_years with most similar avg income
    control_geoids = tracts[tracts['treated'] == 0]['geoid'].unique()
    control_ref_all = df_income[(df_income['geoid'].isin(control_geoids)) & 
                               (df_income['year'].isin(ref_years_all))][['geoid', 'median_household_income_adjusted']]
    
    if control_ref_all.empty:
        print(f"No control income data in pre-period, skipping.")
        continue
    
    # Compute avg income per control tract across pre-period
    control_avg_income = control_ref_all.groupby('geoid')['median_household_income_adjusted'].mean()
    
    # Calculate distance to treated median
    control_dist = np.abs(control_avg_income - treat_median)
    control_dist = control_dist.sort_values()
    
    # Target n_controls: n_treated, but allow ±3 tracts
    n_control_target = n_treated
    n_control_min = max(1, n_treated - 3)
    n_control_max = n_treated + 3
    
    # Select controls within tolerance
    n_control_select = min(n_control_max, len(control_dist))
    selected_control_geoids = control_dist.iloc[:n_control_select].index.tolist()
    
    print(f"Selected {len(selected_control_geoids)} control tracts (target: {n_control_target}±3)")
    print(f"Control cohort avg income (years {ref_years_all}): ${control_avg_income[selected_control_geoids].median():,.0f} (median)")
    
    # =========================================
    # Build 4-panel DiD visualization for this cohort
    # =========================================
    # Select pre and post years for this cohort's analysis window
    # Use a standard window: last year before treatment as pre, 3+ years after as post
    pre_year = max(ref_years_all)  # Latest pre-treatment year
    
    # Find post-treatment years (at least 3 years after treatment)
    post_years_available = [y for y in available_years if y >= treat_year + 3]
    if not post_years_available:
        print(f"No post-treatment years (at least 3 years after treatment), skipping visualization.")
        continue
    post_year = max(post_years_available)  # Use latest available post-treatment year for maximum post-period length
    
    print(f"DiD analysis period: {pre_year} → {post_year}")
    
    # Build DiD dataset for this cohort
    did_data = []
    
    # Treated tracts
    for geoid in treated_geoids:
        pre_income_row = df_income[(df_income['geoid'] == geoid) & (df_income['year'] == pre_year)]
        post_income_row = df_income[(df_income['geoid'] == geoid) & (df_income['year'] == post_year)]
        
        if pre_income_row.empty or post_income_row.empty:
            continue
        
        pre_val = pre_income_row['median_household_income_adjusted'].values[0]
        post_val = post_income_row['median_household_income_adjusted'].values[0]
        income_change = post_val - pre_val
        
        did_data.append({
            'geoid': geoid,
            'treated': 1,
            'pre_income': pre_val,
            'post_income': post_val,
            'income_change': income_change,
            'pct_change': (income_change / pre_val * 100) if pre_val > 0 else np.nan
        })
    
    # Control tracts
    for geoid in selected_control_geoids:
        pre_income_row = df_income[(df_income['geoid'] == geoid) & (df_income['year'] == pre_year)]
        post_income_row = df_income[(df_income['geoid'] == geoid) & (df_income['year'] == post_year)]
        
        if pre_income_row.empty or post_income_row.empty:
            continue
        
        pre_val = pre_income_row['median_household_income_adjusted'].values[0]
        post_val = post_income_row['median_household_income_adjusted'].values[0]
        income_change = post_val - pre_val
        
        did_data.append({
            'geoid': geoid,
            'treated': 0,
            'pre_income': pre_val,
            'post_income': post_val,
            'income_change': income_change,
            'pct_change': (income_change / pre_val * 100) if pre_val > 0 else np.nan
        })
    
    if not did_data:
        print(f"No DiD data collected, skipping.")
        continue
    
    did_df = pd.DataFrame(did_data)
    treated_tracts_df = did_df[did_df['treated'] == 1]
    control_tracts_df = did_df[did_df['treated'] == 0]
    
    treated_mean_change = treated_tracts_df['income_change'].mean()
    control_mean_change = control_tracts_df['income_change'].mean()
    did_estimate = treated_mean_change - control_mean_change
    
    print(f"\nDiD Estimate: ${did_estimate:,.2f}")
    print(f"  Treated mean change: ${treated_mean_change:,.2f}")
    print(f"  Control mean change: ${control_mean_change:,.2f}")
    
    # Plot 4-panel DiD visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cohort {rank}: Treatment Year {treat_year} ({pre_year} → {post_year})\n"
                 f"n_treated={len(treated_tracts_df)}, n_control={len(control_tracts_df)}, "
                 f"DiD Est.=${did_estimate:,.0f}", 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Bar chart of mean changes
    ax = axes[0, 0]
    groups = ['Treated\n(got parks)', 'Control\n(no parks)']
    means = [treated_mean_change, control_mean_change]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(groups, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Mean Income Change ($)', fontsize=11, fontweight='bold')
    ax.set_title('Average Income Change by Treatment Status', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    # Plot 2: Distribution histograms
    ax = axes[0, 1]
    ax.hist(control_tracts_df['income_change'], bins=15, alpha=0.6, label='Control', color='#e74c3c', edgecolor='black')
    ax.hist(treated_tracts_df['income_change'], bins=15, alpha=0.6, label='Treated', color='#2ecc71', edgecolor='black')
    ax.axvline(control_mean_change, color='#c0392b', linestyle='--', linewidth=2.5, label=f'Control Mean')
    ax.axvline(treated_mean_change, color='#27ae60', linestyle='--', linewidth=2.5, label=f'Treated Mean')
    ax.set_xlabel('Income Change ($)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Income Changes', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Box plot
    ax = axes[1, 0]
    box_data = [control_tracts_df['income_change'], treated_tracts_df['income_change']]
    bp = ax.boxplot(box_data, labels=groups, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Income Change ($)', fontsize=11, fontweight='bold')
    ax.set_title('Box Plot of Income Changes', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Scatter plot (pre vs post) with trend lines
    ax = axes[1, 1]
    ax.scatter(control_tracts_df['pre_income'], control_tracts_df['post_income'], 
              alpha=0.5, s=30, label='Control', color='#e74c3c')
    ax.scatter(treated_tracts_df['pre_income'], treated_tracts_df['post_income'], 
              alpha=0.5, s=30, label='Treated', color='#2ecc71')
    
    # Add 45-degree line (no change)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=1, label='No change')
    
    # Add trend lines for treated and control
    if len(treated_tracts_df) > 1:
        z_treated = np.polyfit(treated_tracts_df['pre_income'], treated_tracts_df['post_income'], 1)
        p_treated = np.poly1d(z_treated)
        x_range = np.linspace(treated_tracts_df['pre_income'].min(), treated_tracts_df['pre_income'].max(), 100)
        ax.plot(x_range, p_treated(x_range), linestyle='-', linewidth=2.5, 
               color='#27ae60', alpha=0.8, label='Treated trend')
    
    if len(control_tracts_df) > 1:
        z_control = np.polyfit(control_tracts_df['pre_income'], control_tracts_df['post_income'], 1)
        p_control = np.poly1d(z_control)
        x_range = np.linspace(control_tracts_df['pre_income'].min(), control_tracts_df['pre_income'].max(), 100)
        ax.plot(x_range, p_control(x_range), linestyle='-', linewidth=2.5, 
               color='#c0392b', alpha=0.8, label='Control trend')
    
    ax.set_xlabel(f'Income in {pre_year} ($)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Income in {post_year} ($)', fontsize=11, fontweight='bold')
    ax.set_title('Pre vs Post Period Income (with Trend Lines)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # =========================================
    # Build event-study trends for this cohort
    # =========================================
    print(f"\nBuilding event-study for cohort {rank}...")
    
    treated_trend = []
    control_trend = []
    
    for rel_yr in rel_years:
        actual_yr = treat_year + rel_yr
        if actual_yr not in available_years:
            treated_trend.append(np.nan)
            control_trend.append(np.nan)
            continue
        
        treated_inc = df_income[(df_income['geoid'].isin(treated_geoids)) & 
                               (df_income['year'] == actual_yr)]['median_household_income_adjusted'].mean()
        control_inc = df_income[(df_income['geoid'].isin(selected_control_geoids)) & 
                               (df_income['year'] == actual_yr)]['median_household_income_adjusted'].mean()
        
        treated_trend.append(treated_inc)
        control_trend.append(control_inc)
    
    if all(np.isnan(treated_trend)) or all(np.isnan(control_trend)):
        print(f"No valid income data for event-study, skipping.")
        continue
    
    all_event_study_results.append({
        'rank': rank,
        'treat_year': treat_year,
        'n_treated': n_treated,
        'n_control': len(selected_control_geoids),
        'rel_years': rel_years,
        'treated_trend': treated_trend,
        'control_trend': control_trend
    })

# =========================================
# Plot event-study results
# =========================================
print(f"\n{'='*70}")
print(f"PLOTTED COHORTS: {len(all_event_study_results)}/3")
print(f"{'='*70}\n")

print("Generating event-study plots...")

n_cohorts = len(all_event_study_results)
if n_cohorts > 0:
    fig, axes = plt.subplots(1, n_cohorts, figsize=(6*n_cohorts, 5), squeeze=False)
    axes = axes[0]
    
    for ax, res in zip(axes, all_event_study_results):
        treated_trend = pd.Series(res['treated_trend'], index=res['rel_years'])
        control_trend = pd.Series(res['control_trend'], index=res['rel_years'])
        treat_year = res['treat_year']
        
        ax.plot(treated_trend.index, treated_trend.values, marker='o', label='Treated', 
                color='#2ecc71', linewidth=2.5, markersize=8)
        ax.plot(control_trend.index, control_trend.values, marker='s', label='Control', 
                color='#e74c3c', linewidth=2.5, markersize=8)
        
        # Add linear trend lines for treated and control
        treated_valid = treated_trend.dropna()
        control_valid = control_trend.dropna()
        
        if len(treated_valid) > 1:
            z_treated = np.polyfit(treated_valid.index, treated_valid.values, 1)
            p_treated = np.poly1d(z_treated)
            x_range = np.linspace(treated_valid.index.min(), treated_valid.index.max(), 100)
            ax.plot(x_range, p_treated(x_range), linestyle='--', linewidth=2, 
                   color='#27ae60', alpha=0.6, label='Treated trend')
        
        if len(control_valid) > 1:
            z_control = np.polyfit(control_valid.index, control_valid.values, 1)
            p_control = np.poly1d(z_control)
            x_range = np.linspace(control_valid.index.min(), control_valid.index.max(), 100)
            ax.plot(x_range, p_control(x_range), linestyle='--', linewidth=2, 
                   color='#c0392b', alpha=0.6, label='Control trend')
        
        ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Treatment year (t=0)')
        
        ax.set_xlabel('Years relative to treatment', fontsize=11, fontweight='bold')
        ax.set_ylabel('Median household income (adjusted)', fontsize=11, fontweight='bold')
        ax.set_title(f"Cohort {res['rank']}: Treatment Year {treat_year}\n"
                     f"n_treated={res['n_treated']}, n_control={res['n_control']}", 
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_xticks(rel_years)
    
    plt.tight_layout()
    plt.show()
    
    print("Event-study plots complete!")






