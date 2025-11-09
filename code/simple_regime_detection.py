import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "temporal_files" / "regime_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    print("Loading features data...")
    df = pd.read_csv(DATA_DIR / "features_data.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    print(f"  Loaded {len(df)} days from {df['Date'].min()} to {df['Date'].max()}")
    return df

# VIX-based regime classification
def create_vix_regimes(df):
    print(f"\n{'='*80}")
    print(f"CREATING VIX-BASED REGIME CLASSIFICATION")
    print(f"{'='*80}\n")
    
# VIX-based regime classification
def create_vix_regimes(df):
    print(f"\n{'='*80}")
    print(f"CREATING VIX-BASED REGIME CLASSIFICATION")
    print(f"{'='*80}\n")
    
    regime_df = df[['Date', 'VIX']].copy()
    
    regime_df['regime_dominant'] = 0
    regime_df.loc[df['VIX'] >= 20, 'regime_dominant'] = 1
    regime_df.loc[df['VIX'] >= 30, 'regime_dominant'] = 2
    
    # Soft probabilities using smoothed transitions
    regime_df['regime_0_prob'] = np.clip((25 - df['VIX']) / 5, 0, 1)
    regime_df['regime_2_prob'] = np.clip((df['VIX'] - 25) / 5, 0, 1)
    
    regime_df['regime_1_prob'] = 1 - regime_df['regime_0_prob'] - regime_df['regime_2_prob']
    regime_df['regime_1_prob'] = np.clip(regime_df['regime_1_prob'], 0, 1)
    
    # Normalize probabilities
    prob_sum = (regime_df['regime_0_prob'] + 
                regime_df['regime_1_prob'] + 
                regime_df['regime_2_prob'])
    regime_df['regime_0_prob'] /= prob_sum
    regime_df['regime_1_prob'] /= prob_sum
    regime_df['regime_2_prob'] /= prob_sum
    
    return regime_df

def analyze_regimes(regime_df, df_full):
    print(f"\n{'='*80}")
    print(f"REGIME ANALYSIS")
    print(f"{'='*80}\n")
    
    # Merge with full data
    df = pd.merge(regime_df, df_full, on='Date', how='left')
    
    print("Regime Distribution:")
    regime_counts = regime_df['regime_dominant'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(regime_df)
        print(f"  Regime {regime}: {count:4d} days ({pct:5.1f}%)")
    
    print(f"\n Average VIX by Regime:")
    for regime in [0, 1, 2]:
        mask = (regime_df['regime_dominant'] == regime)
        avg_vix = df.loc[mask, 'VIX_x'].mean()
        min_vix = df.loc[mask, 'VIX_x'].min()
        max_vix = df.loc[mask, 'VIX_x'].max()
        print(f"  Regime {regime}: {avg_vix:.2f} (range: {min_vix:.2f} - {max_vix:.2f})")
    
    print(f"\n Regime Persistence (average consecutive days):")
    # Calculate run lengths
    regime_changes = regime_df['regime_dominant'].diff().ne(0).cumsum()
    run_lengths = regime_df.groupby(regime_changes).size()
    
    for regime in [0, 1, 2]:
        mask = (regime_df['regime_dominant'] == regime)
        regime_runs = regime_df[mask].groupby(regime_changes[mask]).size()
        avg_duration = regime_runs.mean()
        print(f"  Regime {regime}: {avg_duration:.1f} trading days")
    
    # Transition frequencies
    print(f"\n Regime Transitions:")
    transitions = pd.crosstab(
        regime_df['regime_dominant'].shift(), 
        regime_df['regime_dominant'],
        normalize='index'
    )
    transitions.index = [f"From Regime {int(i)}" if not pd.isna(i) else "Start" 
                         for i in transitions.index]
    transitions.columns = [f"→ Regime {int(i)}" for i in transitions.columns]
    print(transitions.to_string())

def save_results(regime_df):
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Save regime probabilities
    output_csv = OUTPUT_DIR / "regime_probabilities.csv"
    regime_df.to_csv(output_csv, index=False)
    print(f"✓ Saved regime probabilities: {output_csv}")
    
    # Save regime statistics
    stats_file = OUTPUT_DIR / "regime_statistics.csv"
    regime_counts = regime_df['regime_dominant'].value_counts().sort_index()
    
    stats_list = []
    for regime, count in regime_counts.items():
        stats_list.append({
            'regime': regime,
            'count': count,
            'percentage': 100 * count / len(regime_df),
            'avg_probability': regime_df[f'regime_{regime}_prob'].mean()
        })
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(stats_file, index=False)
    print(f"✓ Saved regime statistics: {stats_file}")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")

def visualize_regimes(regime_df):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        print(f"\n{'='*80}")
        print(f"CREATING VISUALIZATIONS")
        print(f"{'='*80}\n")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: VIX with regime boundaries
        ax1 = axes[0]
        ax1.plot(regime_df['Date'], regime_df['VIX'], 
                linewidth=1, alpha=0.7, color='black', label='VIX')
        ax1.axhline(y=20, color='orange', linestyle='--', 
                   linewidth=2, label='Medium Vol Threshold')
        ax1.axhline(y=30, color='red', linestyle='--', 
                   linewidth=2, label='High Vol Threshold')
        ax1.fill_between(regime_df['Date'], 0, 20, alpha=0.1, color='green')
        ax1.fill_between(regime_df['Date'], 20, 30, alpha=0.1, color='orange')
        ax1.fill_between(regime_df['Date'], 30, 100, alpha=0.1, color='red')
        ax1.set_ylabel('VIX Level', fontsize=11)
        ax1.set_title('VIX Volatility Index with Regime Boundaries', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regime probabilities
        ax2 = axes[1]
        ax2.plot(regime_df['Date'], regime_df['regime_0_prob'], 
                label='Regime 0 (Low Vol)', color='green', linewidth=1.5, alpha=0.8)
        ax2.plot(regime_df['Date'], regime_df['regime_1_prob'], 
                label='Regime 1 (Medium Vol)', color='orange', linewidth=1.5, alpha=0.8)
        ax2.plot(regime_df['Date'], regime_df['regime_2_prob'], 
                label='Regime 2 (High Vol)', color='red', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('Probability', fontsize=11)
        ax2.set_title('Regime Probabilities (Smoothed)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot 3: Dominant regime
        ax3 = axes[2]
        colors = ['green', 'orange', 'red']
        for regime in [0, 1, 2]:
            mask = (regime_df['regime_dominant'] == regime)
            dates = regime_df.loc[mask, 'Date']
            y_vals = np.ones(len(dates)) * regime
            ax3.scatter(dates, y_vals, c=colors[regime], s=3, 
                       alpha=0.6, label=f'Regime {regime}')
        
        ax3.set_ylabel('Dominant Regime', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_title('Market Regime Classification', fontsize=12, fontweight='bold')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Low Vol', 'Medium Vol', 'High Vol'])
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        plot_file = OUTPUT_DIR / "regime_visualization.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {plot_file}")
        
        plt.close()
        
    except ImportError:
        print("Skipping visualization (matplotlib not available)")
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")

def main():
    print(f"\n{'='*80}")
    print(f"SIMPLE VIX-BASED REGIME DETECTION")
    print(f"{'='*80}\n")
    
    # Load data
    df = load_data()
    
    # Create VIX-based regimes
    regime_df = create_vix_regimes(df)
    
    # Analyze regimes
    analyze_regimes(regime_df, df)
    
    # Save results
    save_results(regime_df)
    
    # Visualize
    visualize_regimes(regime_df)
    
    print(f"\n{'='*80}")
    print(f"REGIME DETECTION COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nRegime Definitions:")
    print(f"  • Regime 0 (Low Vol):    VIX < 20  = Bull market conditions")
    print(f"  • Regime 1 (Medium Vol): 20 ≤ VIX < 30 = Uncertain/Sideways")
    print(f"  • Regime 2 (High Vol):   VIX ≥ 30  = Bear/Crisis conditions")
    print(f"\nNext steps:")
    print(f"  1. Review regime probabilities: {OUTPUT_DIR / 'regime_probabilities.csv'}")
    print(f"  2. Integrate into feature_engineering_v2.py")
    print(f"  3. Create regime-interaction features")
    print()

if __name__ == "__main__":
    main()
