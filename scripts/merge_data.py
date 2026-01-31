import pandas as pd


def main():
    # Load raw datasets
    customers = pd.read_csv("data/raw/customers.csv")
    loans = pd.read_csv("data/raw/loans.csv")
    bureau = pd.read_csv("data/raw/bureau_data.csv")

    # Merge: loans is the main table
    df = loans.merge(customers, on="cust_id", how="left")
    df = df.merge(bureau, on="cust_id", how="left")

    # Save merged dataset
    df.to_csv("data/raw/merged_data.csv", index=False)

    print("Merged dataset created successfully")
    print("Final shape:", df.shape)
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    main()