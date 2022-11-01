import os
import pandas as pd


# Data load
df = pd.read_csv("./data/metadata_rgb_only.csv")


def modify_path(row, normal):
    if normal:
        return pd.Series({'image': os.path.join("healthy", row)})
    else:
        return pd.Series({'image': os.path.join("tumor", row)})


df.update(df.apply(lambda x: modify_path(x[1], x[2] == 'normal'), axis=1))
print(df.head())

# Data splits
# Data with cancer
df_cancer = df.iloc[:2422, [1, 2, 5]].sample(frac=1)
df_cancer_train = df_cancer.sample(frac=0.7)
df_cancer_val = df_cancer.drop(df_cancer_train.index).sample(frac=0.75)
df_cancer_test = df_cancer.drop(
    df_cancer_train.index).drop(df_cancer_val.index)

# Data without cancer
df_not_cancer = df.iloc[2421:, [1, 2, 5]].sample(frac=1)
df_not_cancer_train = df_not_cancer.sample(frac=0.7)
df_not_cancer_val = df_not_cancer.drop(
    df_not_cancer_train.index).sample(frac=0.75)
df_not_cancer_test = df_not_cancer.drop(
    df_not_cancer_train.index).drop(df_not_cancer_val.index)

# Merge datasets
df_train = pd.concat([df_cancer_train, df_not_cancer_train]).sample(frac=1)
df_val = pd.concat([df_cancer_val, df_not_cancer_val]).sample(frac=1)
df_test = pd.concat([df_cancer_test, df_not_cancer_test]).sample(frac=1)

# Create dir
if not os.path.isdir("./data/"):
    os.mkdir("./data/")

# Data output
df_train.to_csv("./data/train.csv")
df_val.to_csv("./data/val.csv")
df_test.to_csv("./data/test.csv")
