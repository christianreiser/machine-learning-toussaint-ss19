import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ts = pd.Series(np.random.randn(1000),
                index=pd.date_range('1/1/2000', periods=1000))


ts = ts.cumsum()

ts.plot()

plt.show()
