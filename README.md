# Synthetic Data Generation with Advanced Validation

A robust pipeline for generating statistically similar synthetic data with comprehensive validation and overfitting risk assessment.

## 📚 Table of Contents

- [Synthetic Data Generation with Advanced Validation](#synthetic-data-generation-with-advanced-validation)
  - [📚 Table of Contents](#-table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Quick Start](#quick-start)
  - [Output example](#output-example)
  - [Understanding Results](#understanding-results)
    - [Statistical Validation Tests](#statistical-validation-tests)
    - [Overfitting Risk Assessment](#overfitting-risk-assessment)
    - [Mixed Results Interpretation](#mixed-results-interpretation)
  - [Understanding Validation Thresholds](#understanding-validation-thresholds)
    - [Statistical Tests (p \> 0.05)](#statistical-tests-p--005)
    - [Binary Classification (\<60%)](#binary-classification-60)
  - [Methodology](#methodology)
    - [1. Parameter Learning](#1-parameter-learning)
    - [2. Controlled Generation](#2-controlled-generation)
    - [3. Multi-Level Validation](#3-multi-level-validation)
  - [Features](#features)
  - [Troubleshooting](#troubleshooting)
    - [Result Variations](#result-variations)
    - [Performance Issues](#performance-issues)
    - [Unexpected Results](#unexpected-results)
  - [📋 Technical Details](#-technical-details)
    - [Requirements](#requirements)
  - [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- Git

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/MarufRayhan/synthetic-data-generator.git
cd synthetic-data-generator

# 2. Create virtual environment (RECOMMENDED)
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run generator
python synthetic_data_generator.py
```

## Output example

```python
================================================================================
SYNTHETIC DATA GENERATION - SUMMARY
================================================================================
DATASET OVERVIEW:
 - Scaled from 500 to 1000 samples (2.0x)
  -  Method: Parameter learning + 2% controlled variation

VALIDATION RESULTS:
 -  Value1 distribution: FAIL (KS p=0.0016)
 -  Value2 distribution: PASS (KS p=0.5639)
 -  Overfitting risk: Low (57.7% distinguishability)

FINAL RECOMMENDATION:
 - The model is not able to distinguish between original and synthetic data - The data can be used for training.

 DELIVERABLES:
  - original_dataset.csv (500 samples)
  -  synthetic_dataset.csv (1000 samples)
  -  Comprehensive validation pipeline
================================================================================
Synthetic data pipeline executed successfully!
```

## Understanding Results

### Statistical Validation Tests

| Test                         | Purpose                  | Success Criteria       |
| ---------------------------- | ------------------------ | ---------------------- |
| **Kolmogorov-Smirnov**       | Distribution similarity  | p-value > 0.05         |
| **Chi-Square**               | Categorical distribution | p-value > 0.05         |
| **Correlation Preservation** | Relationship maintenance | Difference < 0.1       |
| **Moment Analysis**          | Shape characteristics    | Mean/Std diff < 5%/10% |

### Overfitting Risk Assessment

| Distinguishability | Risk Level | Interpretation       |
| ------------------ | ---------- | -------------------- |
| **≤60%**           | Low        | Safe for ML training |
| **60-70%**         | Medium     | Monitor performance  |
| **>70%**           | High       | Improve before use   |

### Mixed Results Interpretation

When statistical tests show mixed results (some pass, some fail), the **binary test** provides the definitive assessment:

- **Low risk (≤60%)**: Synthetic data is practically indistinguishable
- **Statistical differences** may not translate to practical problems

## Understanding Validation Thresholds

### Statistical Tests (p > 0.05)

**Industry standard:** 95% confidence level used in scientific research

- **p < 0.05:** Significant difference detected
- **p > 0.05:** No significant difference (good for synthetic data)

### Binary Classification (<60%)

- **50%:** Random guessing baseline
- **<60%:** Low overfitting risk (safe to use)
- **>70%:** High risk (distinguishable datasets)

## Methodology

### 1. Parameter Learning

The system analyzes the original dataset to extract real statistical patterns:

- Category probability distributions
- Continuous variable parameters (mean, standard deviation)
- Correlation structures between variable

### 2. Controlled Generation

Synthetic data is generated using learned parameters with controlled variation:

- **2% parameter variation** prevents exact copying
- **Maintains statistical similarity** while ensuring uniqueness
- **Preserves relationships** between variables

### 3. Multi-Level Validation

Comprehensive validation using multiple approaches:

- **Statistical tests** for mathematical similarity
- **Machine learning detection** for practical overfitting assessment
- **Combined risk assessment** for clear decision-making

## Features

- **Parameter Learning**: Extracts patterns from actual data instead of using hardcoded values
- **Comprehensive Validation**: Multiple statistical tests plus ML-based overfitting detection
- **Scalable Generation**: Creates larger datasets from smaller originals
- **Risk Assessment**: Binary classification to detect distinguishability

## Troubleshooting

### Result Variations

Results may vary slightly between runs due to:

- **Random seed differences** across Python/NumPy versions
- **Floating-point precision** variations
- **Platform differences** (Windows/Mac/Linux)

This is normal behavior in ML. Focus on:

- **Risk level consistency** (Low/Medium/High)
- **Overall pattern similarity**
- **Final recommendations**

### Performance Issues

For large datasets:

- Reduce `num_samples` if memory issues occur
- Consider running validation on subsets for very large data

### Unexpected Results

If validation shows high risk:

- **Reduce variation parameter** (try 0.01 instead of 0.02)
- **Check data characteristics** (very small datasets may be problematic)
- **Review parameter learning** (ensure sufficient original data)

## 📋 Technical Details

### Requirements

- **Python**: 3.8 or higher
- **Dependencies**: See `requirements.txt`

## License

[MIT](https://choosealicense.com/licenses/mit/)

Copyright (c) 2025 MD Maruf Rayhan
