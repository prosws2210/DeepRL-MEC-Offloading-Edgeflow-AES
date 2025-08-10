# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import time
from scipy.special import lambertw
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

def plot_gain(gain_his, rolling_intv=20):
    gain_array = np.asarray(gain_his)
    df = pd.DataFrame(gain_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(np.arange(len(gain_array)) + 1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(gain_array)) + 1,
                     df.rolling(rolling_intv, min_periods=1).min()[0],
                     df.rolling(rolling_intv, min_periods=1).max()[0],
                     color='b', alpha=0.2)
    plt.ylabel('Gain Ratio')
    plt.xlabel('Learning Steps')
    plt.title('Gain Ratio Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def bisection(h, M, weights=[]):
    # Bisection method for computing optimal resource allocation

    # Constants (hardware/system specific)
    o, p, u = 100, 3, 0.7
    eta1 = ((u * p) ** (1 / 3)) / o
    ki = 1e-26
    eta2 = u * p / 1e-10
    B = 2e6
    Vu = 1.1
    epsilon = B / (Vu * np.log(2))
    
    M0 = np.where(M == 0)[0]
    M1 = np.where(M == 1)[0]

    hi = np.array([h[i] for i in M0])
    hj = np.array([h[i] for i in M1])

    if len(weights) == 0:
        weights = [1.5 if i % 2 == 1 else 1 for i in range(len(M))]
    wi = np.array([weights[i] for i in M0])
    wj = np.array([weights[i] for i in M1])

    def sum_rate(x):
        sum1 = sum(wi * eta1 * (hi / ki) ** (1 / 3) * x[0] ** (1 / 3))
        sum2 = sum(wj[i] * epsilon * x[i + 1] * np.log(1 + eta2 * hj[i] ** 2 * x[0] / x[i + 1])
                   for i in range(len(M1)))
        return sum1 + sum2

    def phi(v, j):
        return 1 / (-1 - 1 / lambertw(-1 / (np.exp(1 + v / wj[j] / epsilon))).real)

    def p1(v):
        return 1 / (1 + eta2 * sum(hj[j] ** 2 * phi(v, j) for j in range(len(M1))))
    
    def Q(v):
        # This part calculates the power needed for local computing at 'price' v
        sum1 = sum(wi * eta1 * (hi / ki) ** (1 / 3)) * p1(v) ** (-2 / 3) / 3
        # This part calculates the power needed for offloading (transmission) at 'price' v
        sum2 = sum(wj[j] * hj[j] ** 2 / (1 + 1 / phi(v, j)) for j in range(len(M1)))
        # Q(v) = (Power for local) + (Power for offloading) - (The price itself)
        return sum1 + sum2 * epsilon * eta2 - v

    def tau(v, j):
        return eta2 * hj[j] ** 2 * p1(v) * phi(v, j)

    # Bisection
    delta = 0.005
    LB, UB = 0, 999999999
    while UB - LB > delta:
        v = (UB + LB) / 2
        
        # Add these two lines for debugging
        q_val = Q(v)
        # print(f"      v={v:.4f}, Q(v)={q_val}, LB={LB:.4f}, UB={UB:.4f}")

        if q_val > 0:
            LB = v
        else:
            UB = v

    x = [p1(v)] + [tau(v, j) for j in range(len(M1))]
    return sum_rate(x), x[0], x[1:]


def cd_method(h):
    N = len(h)
    M0 = np.random.randint(2, size=N)
    gain0, a, Tj = bisection(h, M0)
    
    while True:
        improved = False
        for j in range(N):
            M = np.copy(M0)
            M[j] ^= 1
            gain, _, _ = bisection(h, M)
            if gain > gain0:
                gain0 = gain
                M0 = M
                improved = True
        if not improved:
            break
    return gain0, M0


if __name__ == "__main__":
    # Example channel and mode
    h = np.array([6.06e-6, 1.10e-5, 1.00e-7, 1.22e-6, 1.96e-6,
                  1.71e-6, 5.25e-6, 5.89e-7, 4.08e-6, 2.88e-6])
    M = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    # Run single bisection
    gain, a, Tj = bisection(h, M)
    print("Gain:", gain)
    print("a:", a)
    print("Tj:", Tj)

    # Run CD method
    gain0, M0 = cd_method(h)
    print("Max Gain:", gain0)
    print("Best Mode:", M0)

    # Test all data from MATLAB files
    K_list = [10, 20, 30]
    N = 1000  # samples per file

    for K in K_list:
        print(f"\nTesting K = {K}")
        data = sio.loadmat(f'./data/data_{K}')
        channel = data['input_h']
        gain_target = data['output_obj']

        gain_his = []
        gain_his_ratio = []
        mode_his = []

        start_time = time.time()
        for i in range(N):
            if i % (N // 10) == 0:
                print(f"Progress: {i / N:.0%}")

            h = channel[i, :]
            gain0, M0 = cd_method(h)

            gain_his.append(gain0)
            gain_his_ratio.append(gain0 / gain_target[i][0])
            mode_his.append(M0)

        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per sample: {total_time / N:.5f}s")
        print(f"Gain/Max ratio: {sum(gain_his_ratio) / N:.4f}")

        plot_gain(gain_his_ratio)
