# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import deque
import itertools


def test():
    batch = deque(maxlen=200)
    batch.append(("name1", 0, 10))
    batch.append(("name2", 1, 11))
    batch.append(("name3", 2, 12))
    batch.append(("name4", 3, 13))
    batch.append(("name5", 4, 14))
    batch.append(("name6", 5, 15))
    batch.append(("name7", 6, 16))

    print(batch)

    _, subbatch, _ = zip(*batch)
    subbatch = np.array(subbatch)
    print(subbatch)

if __name__ == "__main__":
    test()