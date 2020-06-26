"""
Timeseries plot with error bands
================================

_thumb: .48, .45

"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns.set(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
print(type(fmri))
print(fmri)
print(type(fmri.to_numpy()[2][0]))
numpy_data = [[1, 2, 'kek'], [3, 4, 'bek']]
print(numpy_data)
df = pd.DataFrame(data=numpy_data, index=range(0, 2), columns=["колонка1", "колонка2", "колонка3"])
print(df)
	
# Plot the responses for different events and regions
sns.lineplot(x="колонка1", y="колонка2",
             hue="колонка3",
             data=df)
plt.show()