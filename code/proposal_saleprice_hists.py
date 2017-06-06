# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:37:23 2017

@author: Celso Araujo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./data/train.csv')
fig, ax = plt.subplots(1, 2)
ax[0].hist(train['SalePrice'])
ax[0].set_title('Histogram of SalePrice')
ax[1].hist(np.log1p(train['SalePrice']))
ax[1].set_title('Histogram of log(SalePrice + 1)')
fig.set_size_inches(18, 5)
fig.savefig('fig/saleprice_hists.png', bbox_inches='tight')