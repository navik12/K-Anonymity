# K-Anonymity Using Randomization, Clustering, and Bottom-Up Generalization

## Overview
This project anonymizes datasets using k-Anonymity while maintaining usability. It employs three algorithms—Randomization, Clustering, and Bottom-Up Generalization—and evaluates results using **Distortion (Cost_MD)** and **Loss (Cost_LM)** metrics.

## Features
- **Randomization**: Adds randomness for quick anonymization.
- **Clustering**: Groups similar records for balanced privacy and utility.
- **Bottom-Up Generalization**: Iteratively generalizes data, ideal for high k-values.
- **Metrics**: Evaluate distortion (Cost_MD) and information loss (Cost_LM).

## Data Generalization Hierarchies (DGH)
Hierarchies are used for generalization:
- **Age**: Grouped into broader ranges (e.g., [20-30]).
- **Education**: Generalized from specific levels to broader categories.
- **Gender**: Simplified to Male and Female.
- **Occupation**: Grouped into White-collar, Blue-collar, etc.

## How to Run
1. Install Python 3.x and required libraries (`numpy`, `matplotlib`).
2. Prepare input files (`adult-hw1.csv`) and DGH files (`age.txt`, `education.txt`, etc.).
3. Run `run.py` to anonymize the data and visualize metrics like Cost_MD, Cost_LM, and runtime.

## Results
- **Randomization**: Fastest but less effective for privacy.
- **Clustering**: Best for low k-values with balanced results.
- **Bottom-Up**: Effective for high k-values with minimal data loss.

## Observations
- Use Clustering for low k-values and Bottom-Up for high k-values.
- Randomization is fast but risks more privacy issues.

## Future Work
- Add differential privacy for stronger guarantees.
- Expand to larger datasets with more attributes.
