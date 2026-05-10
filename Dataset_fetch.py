from datasets import load_dataset

ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

# Convert to pandas
import pandas as pd
df = pd.DataFrame(ds['train'])
print(df.head())
print(df['label'].value_counts())  # 0=Fake, 1=Real
