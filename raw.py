import pandas as pd
import numpy as np

# Create a 10x10 DataFrame with random values
df = pd.DataFrame(np.random.randn(10, 10), columns=[f'Col{i+1}' for i in range(10)])
df
