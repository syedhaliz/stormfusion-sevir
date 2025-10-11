"""
Analyze SEVIR events to find distribution of extreme weather events.
This helps determine if we have enough extreme events in our dataset.
"""
import h5py
import numpy as np
from pathlib import Path
import pandas as pd

# Paths
sevir_vil_path = Path.home() / "My Drive/SEVIR_Data/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5"
catalog_path = Path.home() / "My Drive/SEVIR_Data/CATALOG.csv"

print("Loading SEVIR catalog...")
catalog = pd.read_csv(catalog_path, parse_dates=['time_utc'], low_memory=False)
catalog = catalog[catalog['img_type'] == 'vil']

print(f"Opening {sevir_vil_path}...")
with h5py.File(sevir_vil_path, 'r') as hf:
    vil_data = hf['vil']
    n_events = vil_data.shape[0]
    print(f"Total events in file: {n_events}")

    # VIP thresholds (unnormalized, 0-255 scale)
    thresholds = {
        'light': 16,      # VIP ≥ 16
        'moderate': 74,   # VIP ≥ 74
        'heavy': 133,     # VIP ≥ 133
        'severe': 160,    # VIP ≥ 160
        'extreme': 181,   # VIP ≥ 181
        'hail': 219       # VIP ≥ 219
    }

    results = []

    print("\nAnalyzing events...")
    for i in range(n_events):
        event_data = vil_data[i]  # Shape: (384, 384, 49)

        # Get event ID from catalog using index
        event_id = catalog.iloc[i]['id'] if i < len(catalog) else f"Event_{i}"

        # Calculate max VIP and percentage of pixels exceeding each threshold
        max_vip = event_data.max()

        stats = {
            'event_id': event_id,
            'max_vip': max_vip,
        }

        # Count pixels exceeding each threshold
        total_pixels = event_data.size
        for name, threshold in thresholds.items():
            count = (event_data >= threshold).sum()
            percentage = (count / total_pixels) * 100
            stats[f'{name}_pixels'] = count
            stats[f'{name}_pct'] = percentage

        results.append(stats)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_events} events...")

    print(f"\nProcessed all {n_events} events!")

# Convert to DataFrame
df = pd.DataFrame(results)

# Summary statistics
print("\n" + "="*80)
print("EXTREME EVENT ANALYSIS")
print("="*80)

print(f"\nTotal events: {len(df)}")
print(f"\nMax VIP statistics:")
print(f"  Mean: {df['max_vip'].mean():.1f}")
print(f"  Median: {df['max_vip'].median():.1f}")
print(f"  95th percentile: {df['max_vip'].quantile(0.95):.1f}")
print(f"  Max: {df['max_vip'].max():.1f}")

print("\n" + "-"*80)
print("Events with significant extreme weather:")
print("-"*80)

for name, threshold in thresholds.items():
    # Events where at least 0.1% of pixels exceed threshold
    events_with_extreme = df[df[f'{name}_pct'] > 0.1]
    print(f"{name.capitalize():12s} (VIP≥{threshold:3d}): {len(events_with_extreme):3d} events ({len(events_with_extreme)/len(df)*100:.1f}%)")

print("\n" + "-"*80)
print("Top 20 events by extreme weather intensity:")
print("-"*80)

# Sort by hail pixels (most extreme)
top_extreme = df.nlargest(20, 'hail_pixels')
print(f"\n{'Rank':<6} {'Event ID':<12} {'Max VIP':<10} {'Hail%':<10} {'Extreme%':<10} {'Severe%':<10}")
print("-"*80)
for idx, (rank, row) in enumerate(top_extreme.iterrows(), 1):
    print(f"{rank:<6d} {row['event_id']:<12s} {row['max_vip']:<10.1f} {row['hail_pct']:<10.4f} {row['extreme_pct']:<10.4f} {row['severe_pct']:<10.4f}")

# Save full analysis
output_path = Path("data/samples/sevir_extreme_analysis.csv")
df.to_csv(output_path, index=False)
print(f"\n✓ Saved full analysis to {output_path}")

# Create different dataset splits based on extreme content
print("\n" + "="*80)
print("RECOMMENDED DATASET SPLITS")
print("="*80)

# Strategy 1: Use ALL 541 events
all_events = df['event_id'].tolist()
train_ratio = 0.8
n_train = int(len(all_events) * train_ratio)

# Shuffle with fixed seed for reproducibility
np.random.seed(42)
shuffled = np.random.permutation(all_events)
all_train = shuffled[:n_train].tolist()
all_val = shuffled[n_train:].tolist()

print(f"\n1. ALL EVENTS (maximum data):")
print(f"   Train: {len(all_train)} events")
print(f"   Val: {len(all_val)} events")
print(f"   Total: {len(all_events)} events")

# Strategy 2: Stratified split - ensure extreme events in both splits
# Events with hail pixels > 0.1%
extreme_events = df[df['hail_pct'] > 0.1]['event_id'].tolist()
moderate_events = df[df['hail_pct'] <= 0.1]['event_id'].tolist()

n_extreme_train = int(len(extreme_events) * train_ratio)
n_moderate_train = int(len(moderate_events) * train_ratio)

np.random.seed(42)
shuffled_extreme = np.random.permutation(extreme_events)
shuffled_moderate = np.random.permutation(moderate_events)

stratified_train = list(shuffled_extreme[:n_extreme_train]) + list(shuffled_moderate[:n_moderate_train])
stratified_val = list(shuffled_extreme[n_extreme_train:]) + list(shuffled_moderate[n_moderate_train:])

print(f"\n2. STRATIFIED (balanced extreme events):")
print(f"   Train: {len(stratified_train)} events ({n_extreme_train} extreme, {n_moderate_train} moderate)")
print(f"   Val: {len(stratified_val)} events ({len(shuffled_extreme[n_extreme_train:])} extreme, {len(shuffled_moderate[n_moderate_train:])} moderate)")
print(f"   Total: {len(stratified_train) + len(stratified_val)} events")

# Save the ALL EVENTS split (recommended for diagnosing data vs model issue)
with open('data/samples/all_train_ids.txt', 'w') as f:
    for event_id in all_train:
        f.write(f"{event_id}\n")

with open('data/samples/all_val_ids.txt', 'w') as f:
    for event_id in all_val:
        f.write(f"{event_id}\n")

print(f"\n✓ Saved ALL EVENTS split to:")
print(f"  - data/samples/all_train_ids.txt ({len(all_train)} events)")
print(f"  - data/samples/all_val_ids.txt ({len(all_val)} events)")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nTo isolate data vs. model architecture issue:")
print("1. Train on ALL 541 events (432 train / 109 val)")
print("2. If extreme event performance improves → data issue")
print("3. If extreme event performance still poor → model architecture issue")
print("\nNext step: Update Stage04 notebook to use all_train_ids.txt and all_val_ids.txt")
