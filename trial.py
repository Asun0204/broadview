#!/usr/bin/env python
# coding=utf-8
# Created by Asun on 2017/7/5
# Description:

import numpy as np

print np.arange(5)[:, np.newaxis]
print np.arange(5)[:, np.newaxis].shape
print np.arange(5)[np.newaxis, :, np.newaxis]
print np.arange(5)[np.newaxis, :, np.newaxis].shape