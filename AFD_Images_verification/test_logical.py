import numpy as np
import time

if __name__ == '__main__':
    t1 = time.time()
    a = np.random.choice([True, False], size=100000000, p=[0.5, 0.5])
    b = np.random.choice([True, False], size=100000000, p=[0.5, 0.5])
    res = np.sum(np.logical_and(a, b))
    print(res)
    print(time.time() - t1)
    a = [1,3,4, 50, 100, 20]
    print(np.less(a, 20))
