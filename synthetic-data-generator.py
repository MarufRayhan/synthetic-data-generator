import numpy as np
import pandas as pd
import scipy
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
from scipy import stats as scipy_stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def generate_original_dataset():
    print("Generating original dataset...")
    np.random.seed(42)
    num_samples = 500
    
    df = pd.DataFrame(
        {
        "Category1": np.random.choice(["A", "B", "C", "D", "E"],
                                        num_samples,
                                        p=[0.2, 0.4, 0.2, 0.1, 0.1]),
        "Value1": np.random.normal(10, 2, num_samples), 
        "Value2": np.random.normal(20, 6, num_samples), 
        }
    )
    
    print(f"Original dataset generated with 500 samples.", df.shape)
    print(f"Columns: {list(df.columns)}")
    print(f"Sample size: {len(df)}")

    return df

original_data = generate_original_dataset()

def analyze_dataset(data):
    print("\nAnalyzing dataset...")
    category_counts = data['Category1'].value_counts().sort_index()
    print("\nCategory Counts:")
    print(category_counts)
    
    category_probabilities = data['Category1'].value_counts(normalize=True).sort_index()
    print("\nCategory Probabilities:")
    print(category_probabilities)
    
    for col in ['Value1', 'Value2']:
        mean = data[col].mean()
        std_dev = data[col].std()
        min_val = data[col].min()
        max_val = data[col].max()
        print(f"\n{col} - Mean: {mean:.2f}, Std Dev: {std_dev:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}")
    
    corelation = data["Value1"].corr(data["Value2"])
    print(f"\nCorrelation between Value1 and Value2: {corelation:.2f}")
    
    learned_patterns = {
        'sample_size': len(data),
        'categories': list(category_probabilities.index),
        'category_probabilities': category_probabilities.to_dict(),
        'mean_value1': data['Value1'].mean(),
        'std_dev_value1': data['Value1'].std(),
        'mean_value2': data['Value2'].mean(),
        'std_dev_value2': data['Value2'].std(),
        'correlation_value1_value2': corelation
    }
    print("\nLearned Patterns:")
    print(learned_patterns)
    
    return learned_patterns
        
learned_patterns = analyze_dataset(original_data)


def generate_synthetic_dataset(num_samples=1000, learned_patterns=None):
    if learned_patterns is None:
        raise ValueError("Learned patterns must be provided to generate synthetic dataset.")
    
    print("\nGenerating synthetic dataset...")
    np.random.seed(1024)
    
    categories = learned_patterns['categories']
    category_probabilities = [learned_patterns['category_probabilities'][cat] for cat in categories] 
    
    synthetic_dataset_categories = np.random.choice(categories, num_samples, p=category_probabilities)
    
    print(f"Print first 10 categories: {synthetic_dataset_categories[:10]}")
    
    print(f"Using variation of 2% for continuous values")
    variation = .02
    
    mean_value1 = learned_patterns['mean_value1'] * (1 + np.random.normal(0, variation))
    mean_value2 = learned_patterns['mean_value2'] * (1 + np.random.normal(0, variation))
    std_dev_value1 = learned_patterns['std_dev_value1'] * (1 + np.random.normal(0, variation))
    std_dev_value2 = learned_patterns['std_dev_value2'] * (1 + np.random.normal(0, variation))
    
    syntetic_dataset_value1 = np.random.normal(mean_value1, std_dev_value1, num_samples)
    syntetic_dataset_value2 = np.random.normal(mean_value2, std_dev_value2, num_samples)
    
    print(f'Generating synthetic dataset continuous values')
    print(f"Mean Value1: {mean_value1:.2f}, Std Dev Value1: {std_dev_value1:.2f}")
    print(f"Mean Value2: {mean_value2:.2f}, Std Dev Value2: {std_dev_value2:.2f}")
    
    syntetic_dataset = pd.DataFrame({
        'Category1': synthetic_dataset_categories,
        'Value1': syntetic_dataset_value1,
        'Value2': syntetic_dataset_value2
    })
    
    print(f"Synthetic dataset generated with {num_samples} samples.", syntetic_dataset.shape)
    return syntetic_dataset 

syntetic_dataset = generate_synthetic_dataset(num_samples=1000, learned_patterns=learned_patterns) 

def statistical_validation(original_data, synthetic_data):
    
    print("\n Running Statistical Validation...")
    print("\n Doing Kolmogorov-Smirnovt for (KS Test)...")
    
    results = {}
    for col in ['Value1', 'Value2']:
        ks_statistic, ks_p_value = ks_2samp(original_data[col], synthetic_data[col])
        is_similar = ks_p_value > 0.05
        
        results[f'{col}_ks_statistic'] = ks_statistic
        results[f'{col}_ks_p_value'] = ks_p_value
        results[f'{col}_is_similar'] = is_similar
        
        print(f"{col} - KS Statistic: {ks_statistic:.4f}, KS P-Value: {ks_p_value:.4f}, Similar: {is_similar}")
       
    print("\n Doing Chi-Squared Test for Categorical Data...")
    
    original_counts = original_data['Category1'].value_counts().sort_index()
    syntheteic_count = synthetic_data['Category1'].value_counts().reindex(original_counts.index, fill_value=0)
    chi2_stat, chi2_p, val, val2 = chi2_contingency([original_counts, syntheteic_count])
    chi2_is_similar = chi2_p > 0.05
    
    results['category_chi2_stat'] = chi2_stat
    results['category_chi2_pvalue'] = chi2_p
    results['category_similar'] = chi2_is_similar
    
    print(f"Chi-Squared Statistic: {chi2_stat:.4f}, Chi-Squared P-Value: {chi2_p:.4f}, Similar: {chi2_is_similar}")

    for cat in original_counts.index:
        orig_pct = (original_counts[cat] / len(original_data)) * 100
        synth_pct = (syntheteic_count[cat] / len(synthetic_data)) * 100
        diff = abs(orig_pct - synth_pct)
        
        print(f"{cat}: {orig_pct:.1f}% -> {synth_pct:.1f}% (diff: {diff:.1f}%)")
    

    print("\n Doing Moement comparison (mean, std, skewness, kurtosis)...")
    
    for col in ['Value1', 'Value2']:
        mean_orig = original_data[col].mean()
        std_orig = original_data[col].std()
        skew_orig = original_data[col].skew()
        kurtosis_orig = original_data[col].kurtosis()
        
        mean_synth = synthetic_data[col].mean()
        std_synth = synthetic_data[col].std()
        skew_synth = synthetic_data[col].skew()
        kurtosis_synth = synthetic_data[col].kurtosis()
        
        mean_diff_pct = abs(mean_orig - mean_synth) / abs(mean_orig) * 100
        std_diff_pct = abs(std_orig - std_synth) / std_orig * 100
        skew_diff = abs(skew_orig - skew_synth)
        kurt_diff = abs(kurtosis_orig - kurtosis_synth)
        
        results[f'{col}_mean_diff_pct'] = mean_diff_pct
        results[f'{col}_std_diff_pct'] = std_diff_pct
        results[f'{col}_skew_diff'] = skew_diff
        results[f'{col}_kurt_diff'] = kurt_diff
        
        print(f"{col}:")
        print(f"Mean:{mean_orig:.3f}, {mean_synth:.3f} ({mean_diff_pct:.1f}% diff)")
        print(f"Std:{std_orig:.3f}, {std_synth:.3f} ({std_diff_pct:.1f}% diff)")
        print(f"Skewness:{skew_orig:.3f}, {skew_synth:.3f} (diff: {skew_diff:.3f})")
        print(f"Kurtosis:{kurtosis_orig:.3f}, {kurtosis_synth:.3f} (diff: {kurt_diff:.3f})")
        
    
    print("\n Doing Corelation preservation...")
    original_corr = original_data['Value1'].corr(original_data['Value2'])
    synthetic_corr = synthetic_data['Value1'].corr(synthetic_data['Value2'])
    cor_diff = abs(original_corr - synthetic_corr)
    
    results['correlation_original'] = original_corr
    results['correlation_synthetic'] = synthetic_corr
    results['correlation_difference'] = cor_diff < .1
    
    print(f"Original Correlation: {original_corr:.4f}, Synthetic Correlation: {synthetic_corr:.4f}, Difference: {cor_diff:.4f}")
    print(f"Preserved: {'Yes' if cor_diff < 0.1 else 'No'} (diff < 0.1)")
    
    
statisical_result = statistical_validation(original_data, syntetic_dataset)

def execute_binary_classification(original_data, synthetic_data):
    print("\nExecuting Binary Classification...")
    print("Original Data: label = 0, Synthetic Data: label = 1")
    
    original_labled = original_data.copy()
    original_labled['is_synthetic'] = 0
    
    synthetic_labled = synthetic_data.copy()
    synthetic_labled['is_synthetic'] = 1
    
    combined_data = pd.concat([original_labled, synthetic_labled], ignore_index=True)
    
    print(f"Combined dataset shape: {combined_data.shape}")
    
    le = LabelEncoder()
    features = pd.DataFrame({
        'Category1_encoded': le.fit_transform(combined_data['Category1']),
        'Value1': combined_data['Value1'],
        'Value2': combined_data['Value2']
    })
    
    target = combined_data['is_synthetic']
    print(f"   Features: {list(features.columns)}")
    print(f"   Target: is_synthetic (0=original, 1=synthetic)")
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model trained on {len(X_train)} samples")
    print(f"Tested on {len(X_test)} samples")   
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"AUC: {auc:.3f}")
    
    if accuracy <= 0.6:
        risk_level = "Low"
        interpretation = "The model is not able to distinguish between original and synthetic data - The data can be used for training."
    elif accuracy <= 0.7:
        risk_level = "Medium"
        interpretation = "The model can distinguish between original and synthetic data with some confidence - The data can be used for training with caution."
    else:
        risk_level = "High"
        interpretation = "The model can distinguish between original and synthetic data with high confidence - The data should not be used for training."
        
    print(f"\nRisk Level: {risk_level}")
    print(f"Interpretation: {interpretation}")
    return {
        'accuracy': accuracy,
        'auc': auc,
        'risk_level': risk_level,
        'interpretation': interpretation
    }
    
binary_result = execute_binary_classification(original_data, syntetic_dataset)

def generate_final_summary():
    """Summary of complete analysis"""
    print("\n" + "="*80)
    print("SYNTHETIC DATA GENERATION - SUMMARY")
    print("="*80)
    
    # Key metrics
    v1_ks_stat, v1_ks_p = ks_2samp(original_data['Value1'], syntetic_dataset['Value1'])
    v2_ks_stat, v2_ks_p = ks_2samp(original_data['Value2'], syntetic_dataset['Value2'])
    
    print(f"DATASET OVERVIEW:")
    print(f" - Scaled from {len(original_data)} to {len(syntetic_dataset)} samples ({len(syntetic_dataset)/len(original_data):.1f}x)")
    print(f"  -  Method: Parameter learning + 2% controlled variation")
    
    print(f"\nVALIDATION RESULTS:")
    print(f" -  Value1 distribution: {'PASS' if v1_ks_p > 0.05 else 'FAIL'} (KS p={v1_ks_p:.4f})")
    print(f" -  Value2 distribution: {'PASS' if v2_ks_p > 0.05 else 'FAIL'} (KS p={v2_ks_p:.4f})")
    print(f" -  Overfitting risk: {binary_result['risk_level']} ({binary_result['accuracy']:.1%} distinguishability)")
    
    print(f"\nFINAL RECOMMENDATION:")
    print(f" - {binary_result['interpretation']}")
    
    print(f"\n DELIVERABLES:")
    print(f"  - original_dataset.csv ({len(original_data)} samples)")
    print(f"  -  synthetic_dataset.csv ({len(syntetic_dataset)} samples)")
    print(f"  -  Comprehensive validation pipeline")
    
    print("="*80)
    
generate_final_summary()

if __name__ == "__main__":
    print("Synthetic data pipeline executed successfully!")