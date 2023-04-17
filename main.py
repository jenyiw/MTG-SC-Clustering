# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:48:34 2023

@author: Jenyi
"""

import tempdataFunctions as tF
import plotumapFunctions as puF

dataset, labels = tF.create_random_data(1000)
puF.plot_umap(dataset, labels, cell_names=['a', 'b', 'c', 'd', 'e'])