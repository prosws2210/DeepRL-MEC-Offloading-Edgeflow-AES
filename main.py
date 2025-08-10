import os
# This will hide the numerous TensorFlow deprecation warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# This will hide the Protobuf version mismatch warnings.
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# -------------------------

import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

from agent import MemoryDNN
from optimizer import bisection


def plot_rate(rate_his, rolling_intv=50):
    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    # The 'seaborn' style is deprecated in modern matplotlib, using 'seaborn-v0_8' instead
    try:
        mpl.style.use('seaborn-v0_8-darkgrid')
    except:
        mpl.style.use('seaborn-darkgrid')
        
    fig, ax = plt.subplots(figsize=(15, 8))

    plt.plot(np.arange(len(rate_array)) + 1,
             np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array)) + 1,
                     np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values),
                     np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values),
                     color='b', alpha=0.2)

    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.title('Performance Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_to_txt(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            # Handle cases where item might be a numpy array
            if isinstance(item, np.ndarray):
                # Convert array to a string for saving
                item_str = ' '.join(map(str, item.flatten()))
                f.write(f"{item_str}\n")
            else:
                f.write(f"{item}\n")


if __name__ == "__main__":

    # Parameters
    N = 10                  # Number of users
    n = 30000               # Number of time frames
    K = N                   # Initial K
    decoder_mode = 'OP'     # 'OP' or 'KNN'
    Memory = 1024           # Memory capacity
    Delta = 32              # Interval for adaptive K

    print(f"#user = {N}, #channel = {n}, K = {K}, decoder = {decoder_mode}, Memory = {Memory}, Delta = {Delta}")

    # Load and scale data
    data = sio.loadmat(f'./data/data_{N}')
    channel = data['input_h'] * 1e6
    rate = data['output_obj']  # This is a (30000, 1) array

    # Split into training and test sets
    split_idx = int(0.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(0.8 * n))

    # Initialize Memory DNN
    mem = MemoryDNN(
        net=[N, 120, 80, N],
        learning_rate=0.01,
        training_interval=10,
        batch_size=128,
        memory_size=Memory
    )

    # Tracking variables
    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []

    start_time = time.time()

    for i in range(n):
        if i % (n // 10) == 0:
            print(f"{i/n:.1%}")

        if i > 0 and i % Delta == 0:
            max_k = max(k_idx_his[-Delta:]) + 1 if Delta > 1 else k_idx_his[-1] + 1
            K = min(max_k + 1, N)

        i_idx = i % split_idx if i < n - num_test else i - n + num_test + split_idx
        h = channel[i_idx, :]

        m_list = mem.decode(h, K, decoder_mode)

        r_list = [bisection(h / 1e6, m)[0] for m in m_list]

        best_idx = np.argmax(r_list)
        best_mode = m_list[best_idx]

        mem.encode(h, best_mode)

        # Save statistics
        rate_his.append(r_list[best_idx])
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0]) 
        k_idx_his.append(best_idx)
        K_his.append(K)
        mode_his.append(best_mode)

    total_time = time.time() - start_time

    # Plot and results
    mem.plot_cost()
    plot_rate(rate_his_ratio)

    # Ensure num_test is not zero to avoid DivisionByZeroError
    if num_test > 0:
        avg_rate = sum(rate_his_ratio[-num_test:]) / num_test
        print(f"Averaged normalized computation rate: {avg_rate:.4f}")
    else:
        print("No test data was processed, cannot calculate average rate.")
        
    print(f"Total time consumed: {total_time:.2f}s")
    print(f"Average time per channel: {total_time / n:.6f}s")

    # --- ADDED: SAVE THE TRAINED MODEL ---
    # Note: Ensure you have created a 'saved_model' directory in your project folder first.
    model_path = "./saved_model/model.ckpt"
    mem.save_model(model_path)
    # ------------------------------------

    # Save data
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")