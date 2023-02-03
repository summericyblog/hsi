import pandas as pd
import numpy as np

a = pd.DataFrame({'a': [1,2,3,np.nan,4,5]})
a=a['a']
b = a.rolling(2, min_periods=1)
c = b.apply(np.nanmean)

for i in b:
    print(i)
    print(np.nanmean(i))
print(c)