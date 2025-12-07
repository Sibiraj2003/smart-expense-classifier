import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data', 'Credit card transactions.csv')
plot_path = os.path.join(project_root, 'ml', 'output', 'plots')

plt_1 = os.path.join(plot_path, 'amtdistribution.png')
plt_2 = os.path.join(plot_path, 'outlier.png')
plt_3 = os.path.join(plot_path, 'expensetype.png')

df = pd.read_csv(data_path)

def run_basic_eda(df):
    """
    Performs initial exploratory data analysis on the dataset.
    This is structured as a production-ready reusable function.

    Args:
        df: Pandas Data Frame
    
    Returns:
        pd.core.dataframe
    """

    print("\nRunning Basic EDA...")
    print("\nDataset Loaded Successfully!")

    # =============================
    # 1. Structure Overview
    # =============================
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("DATASET STRUCTURE")
    print("━━━━━━━━━━━━━━━━━━━━━━")
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)

    # =============================
    # 2. Missing Values
    # =============================
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("MISSING VALUES")
    print("━━━━━━━━━━━━━━━━━━━━━━")
    print(df.isnull().sum())

    # =============================
    # 3. Summary Statistics
    # =============================
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("SUMMARY STATISTICS")
    print("━━━━━━━━━━━━━━━━━━━━━━")
    print(df.describe(include='all'))

    # =============================
    # 4. Value Counts (categorical)
    # =============================
    categorical_cols = df.select_dtypes(include='object').columns

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("CATEGORY DISTRIBUTIONS")
    print("━━━━━━━━━━━━━━━━━━━━━━")

    for col in categorical_cols:
        print(f"\n {col} distribution:")
        print(df[col].value_counts().head(10))

    # =============================
    # 5. Visual Plots
    # =============================
    print("\nGenerating Visual Plots...")

    # Amount Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Amount'], kde=True)
    plt.title("Amount Distribution")
    plt.savefig(plt_1)

    # Boxplot for Outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['Amount'])
    plt.title("Outlier Detection - Amount")
    plt.savefig(plt_2)

    # Expense Type count
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Exp Type')
    plt.title("Expense Type Frequency")
    plt.xticks(rotation=45)
    plt.savefig(plt_3)

    print("\nEDA COMPLETED SUCCESSFULLY!")
    print(type(df))
    return df

if __name__ == '__main__':
    data = run_basic_eda(df)
    print(data)