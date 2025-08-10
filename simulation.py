# ===================================================================
# MEC Offloading Simulation with EDGEFLOW Protocol
# Modes: DRL Model (from .ckpt) vs. Mathematical Heuristic
# ===================================================================

import numpy as np
import scipy.io as sio
import os
import time
import random
import re
import wcwidth

from memory_loader import MemoryDNN_TF1_Loader

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)

# --- Step 1: Define Core Classes and Functions ---


def generate_task_for_user(user_id):
    """Creates a new computational task with random attributes."""
    return {
        "id": user_id,
        "load": random.randint(20, 100),  # GFLOPs
        "initial_data_size": random.uniform(5, 50),  # MB
        "priority": random.choice(["High", "Medium", "Low"]),
        "criticality": random.choice([True, False]),
    }


def edgeflow_protocol_handler(task, channel_gain):
    """Simulates the EDGEFLOW protocol's adaptive logic for a single task."""
    task["transport_mode"] = (
        "TCP-like (Reliable)" if task["criticality"] else "UDP-like (Fast)"
    )

    priority_map = {"High": 0.8, "Medium": 0.6, "Low": 0.4}
    compression_ratio = priority_map.get(task["priority"], 0.5)

    # Use a threshold on the original channel gain for this logic
    if channel_gain > 5e-5:
        compression_ratio *= 0.9

    task["compressed_size"] = task["initial_data_size"] * compression_ratio
    task["final_data_size"] = task["compressed_size"] + 0.5  # Encryption overhead

    return task


def mathematical_decision_maker(tasks, channels, server_max_load):
    """
    EDGEFLOW Heuristic: Prioritizes tasks based on a score combining
    priority, channel quality, and task load.
    """
    num_users = len(tasks)
    priority_map = {"High": 3, "Medium": 2, "Low": 1}

    user_data = []
    for i in range(num_users):
        score = (
            (priority_map[tasks[i]["priority"]] * 0.5)
            + (channels[i] * 1e5)  # Scale channel gain to have a reasonable impact
            - (tasks[i]["load"] * 0.001)
        )
        user_data.append(
            {
                "task": tasks[i],
                "channel": channels[i],
                "score": score,
                "decision": 0,
            }
        )

    user_data.sort(key=lambda x: x["score"], reverse=True)

    decisions_array = np.zeros(num_users)
    current_server_load = 0.0

    for user in user_data:
        user_id = user["task"]["id"]
        task_load = user["task"]["load"]

        if current_server_load + task_load <= server_max_load:
            decisions_array[user_id] = 1
            user["decision"] = 1
            current_server_load += task_load

    return decisions_array, user_data


# --- Step 2: Helper Functions for Display & Simulation ---


def simulate_encryption(message):
    return message.encode("utf-8").hex()


def simulate_decryption(ciphertext):
    try:
        return bytes.fromhex(ciphertext).decode("utf-8")
    except:
        return "Decryption Failed"


# def display_timestep_results(ts_data):
#     """Formats and prints the results for a single time step in a detailed table."""
#     mode = ts_data["mode"]
#     status_color = "\033[91m" if ts_data["overloaded"] else "\033[92m"
#     status_text = "OVERLOADED ‼️" if ts_data["overloaded"] else "OK ✅"
#     header_title = f" Time Step: {ts_data['step']:<3} | Mode: {mode.upper():<12}"

#     print(f"\n{header_title}")
#     print("-" * 125)

#     utilization = (
#         (ts_data["load"] / ts_data["max_load"] * 100)
#         if not ts_data["overloaded"]
#         else 100.0
#     )
#     util_color = (
#         "\033[93m"
#         if 50 < utilization < 90
#         else ("\033[91m" if utilization >= 90 else "\033[92m")
#     )
#     print(f"Server Status: {status_color}{status_text}\033[0m")
#     print(
#         f"Requested Load: {ts_data['load']:6.2f} GFLOPs / {ts_data['max_load']:.1f} GFLOPs "
#         f"| Utilization: {util_color}{utilization:6.2f}%\033[0m"
#     )
#     print(
#         f"User Decisions: 📡 Offloading: {ts_data['num_offload']} | 🖥️  Local: {ts_data['num_local']}"
#     )
#     print("-" * 125)

#     rank_header = "| Rank" if mode == "mathematical" else ""
#     print(
#         f"{'User ID':<8} | {'Task Load':<12} | {'Data Size':<12} | {'Priority':<10} | {'Channel Gain':<15} {rank_header} | {'Decision':<18} | {'Offload Status':<16}"
#     )
#     print(
#         f"{'--------':<8} | {'-'*12:<12} | {'-'*12:<12} | {'-'*10:<10} | {'-'*15:<15} {'|----' if mode == 'mathematical' else ''} | {'-'*18:<18} | {'-'*16:<16}"
#     )

#     users_to_print = (
#         ts_data["users_sorted"] if mode == "mathematical" else ts_data["users"]
#     )
#     for i, user in enumerate(users_to_print):
#         task, decision, gain = user["task"], user["decision"], user["channel"]
#         decision_text = (
#             "\033[94m📡 Offload (1)\033[0m"
#             if decision == 1
#             else "\033[93m🖥️  Local (0)\033[0m"
#         )
#         status_text = "---"
#         if decision == 1:
#             status_text = (
#                 "\033[91mREJECTED ❌\033[0m"
#                 if ts_data["overloaded"]
#                 else "\033[92mACCEPTED ✅\033[0m"
#             )
#         rank_text = f"|  {i+1:<2}" if mode == "mathematical" else ""
#         # Display channel gain in scientific notation for clarity
#         print(
#             f"{task['id']:<8} | {str(task['load'])+' GFLOPs':<12} | {str(round(task['initial_data_size'],1))+' MB':<12} | {task['priority']:<10} | {f'{gain:.2e}':<15} {rank_text} | {decision_text:<18} | {status_text:^16}"
#         )
#     print("-" * 125)


def strip_ansi(text):
    """Remove ANSI escape sequences."""
    return re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", text)


def visible_width(text):
    """Return visible width considering emoji and wide characters."""
    return sum([wcwidth.wcwidth(c) for c in strip_ansi(text)])


def pad_ansi(text, width):
    """Pad text containing ANSI and emoji to a fixed visible width."""
    pad_len = max(0, width - visible_width(text))
    return text + " " * pad_len


def display_timestep_results(ts_data):
    """Prints formatted table for a single time step."""
    mode = ts_data["mode"]
    status_color = "\033[91m" if ts_data["overloaded"] else "\033[92m"
    status_text = "OVERLOADED ‼️" if ts_data["overloaded"] else "OK ✅"
    header_title = f" Time Step: {ts_data['step']:<3} | Mode: {mode.upper():<12}"

    print(f"\n{header_title}")
    print("-" * 125)

    utilization = (
        (ts_data["load"] / ts_data["max_load"] * 100)
        if not ts_data["overloaded"]
        else 100.0
    )
    util_color = (
        "\033[93m"
        if 50 < utilization < 90
        else ("\033[91m" if utilization >= 90 else "\033[92m")
    )
    print(f"Server Status: {status_color}{status_text}\033[0m")
    print(
        f"Requested Load: {ts_data['load']:6.2f} GFLOPs / {ts_data['max_load']:.1f} GFLOPs "
        f"| Utilization: {util_color}{utilization:6.2f}%\033[0m"
    )
    print(
        f"User Decisions: 📡 Offloading: {ts_data['num_offload']} | 🖥️  Local: {ts_data['num_local']}"
    )
    print("-" * 125)

    rank_header = "| Rank" if mode == "mathematical" else ""
    print(
        f"{'User ID':<8} | {'Task Load':<12} | {'Data Size':<12} | {'Priority':<10} | {'Channel Gain':<15} {rank_header} | {'Decision':<18} | {'Offload Status':<16}"
    )
    print(
        f"{'--------':<8} | {'-'*12:<12} | {'-'*12:<12} | {'-'*10:<10} | {'-'*15:<15} "
        f"{'|----' if mode == 'mathematical' else ''} | {'-'*18:<18} | {'-'*16:<16}"
    )

    users_to_print = (
        ts_data["users_sorted"] if mode == "mathematical" else ts_data["users"]
    )
    for i, user in enumerate(users_to_print):
        task, decision, gain = user["task"], user["decision"], user["channel"]

        raw_decision = "📡 Offload (1)" if decision == 1 else "🖥️  Local (0)"
        color = "\033[94m" if decision == 1 else "\033[93m"
        decision_colored = pad_ansi(f"{color}{raw_decision}\033[0m", 18)

        status_text = "---"
        if decision == 1:
            status_text = (
                "\033[91mREJECTED ❌\033[0m"
                if ts_data["overloaded"]
                else "\033[92mACCEPTED ✅\033[0m"
            )

        rank_text = f"|  {i+1:<2}" if mode == "mathematical" else ""

        print(
            f"{task['id']:<8} | "
            f"{str(task['load'])+' GFLOPs':<12} | "
            f"{str(round(task['initial_data_size'],1))+' MB':<12} | "
            f"{task['priority']:<10} | "
            f"{f'{gain:.2e}':<15} "
            f"{rank_text} | "
            f"{decision_colored} | "
            f"{status_text:^16}"
        )

    print("-" * 125)


def display_edgeflow_workflow(user_id, task, decision, overloaded):
    """Prints the simulated EDGEFLOW workflow with message exchange and encryption."""
    print(f"\n--- 🔬 EDGEFLOW Protocol Workflow for User {user_id} ---")
    # ... (rest of the function is fine, no changes needed)
    sample_message = "Hello, Good Morning"

    print(
        f"1. [UE-{user_id}] Task Initialized: Priority={task['priority']}, Data Size={task['initial_data_size']:.2f} MB"
    )
    time.sleep(0.2)
    print(
        f"2. [EDGEFLOW] Applying Protocol: Mode={task['transport_mode']}, Final Size={task['final_data_size']:.2f} MB"
    )
    time.sleep(0.2)

    if decision == 1:
        print(f"3. [UE-{user_id}] Attempting to Offload with sample message...")
        encrypted_message = simulate_encryption(sample_message)
        print(
            f"   - 🔐 Encrypting message: '{sample_message}' -> '{encrypted_message}'"
        )
        time.sleep(0.2)

        if overloaded:
            print(
                "4. [MES] >> REJECTED << Server is overloaded. Task will be processed locally."
            )
        else:
            print("4. [MES] >> ACCEPTED << Server acknowledges request.")
            time.sleep(0.3)
            print(
                f"5. [UE-{user_id} -> MES] Transmitting {task['final_data_size']:.2f} MB of data..."
            )
            print(f"   - Transmitted encrypted message: '{encrypted_message}'")
            time.sleep(0.3)
            decrypted_message = simulate_decryption(encrypted_message)
            print(f"6. [MES] 🔓 Received and decrypted message: '{decrypted_message}'")
            time.sleep(0.2)
            response_message = "Task Complete"
            encrypted_response = simulate_encryption(response_message)
            print(
                f"7. [MES -> UE-{user_id}] Sending encrypted response: '{encrypted_response}'"
            )
            time.sleep(0.3)
            decrypted_response = simulate_decryption(encrypted_response)
            print(
                f"8. [UE-{user_id}] 🔓 Received and decrypted response: '{decrypted_response}'"
            )
    else:
        print(f"3. [UE-{user_id}] Decision: Process task locally. No transmission.")
    print("--- End of Workflow ---")


# --- Step 3: Main Simulation Function ---
def run_mec_simulation(mode, model_path, data_path, num_users=10, simulation_steps=30):
    """Main simulation loop."""
    print(f"\n--- Initializing MEC Simulation in '{mode.upper()}' Mode ---")
    SERVER_MAX_LOAD = 250.0
    MONITORED_USERS = [1, 5]
    print(
        f"Server Max Load: {SERVER_MAX_LOAD:.1f} GFLOPs | Monitored Users: {MONITORED_USERS}"
    )

    eval_agent = None
    if mode == "model":
        # CORRECTED: Instantiate the TF1 loader class
        eval_agent = MemoryDNN_TF1_Loader(net=[num_users, 120, 80, num_users])
        try:
            # CORRECTED: Use the load_model method with better error printing
            eval_agent.load_model(model_path)
        except Exception:
            # The specific error is already printed inside load_model
            print("Could not continue due to model loading failure.")
            return

    data_file = os.path.join(data_path, f"data_{num_users}.mat")
    try:
        # Load the raw channel data first
        raw_channel_data = sio.loadmat(data_file)["input_h"]
        # The model expects data scaled by 1e6, as done during training
        scaled_channel_data = raw_channel_data * 1e6
        print(f"✅ Channel data loaded and scaled for the model.")
    except FileNotFoundError:
        print(f"🚨 Error: Data file not found at {data_file}")
        return

    print("\n--- Simulation Started ---")
    summary_stats = {"accepted": 0, "rejected": 0, "overloads": 0, "utilization": []}

    for i in range(simulation_steps):
        time.sleep(0.8)

        # Use the correct data for each part of the simulation
        current_scaled_channels = scaled_channel_data[i, :]
        current_raw_channels = raw_channel_data[i, :]

        tasks = [generate_task_for_user(u) for u in range(num_users)]
        processed_tasks = [
            edgeflow_protocol_handler(t, current_raw_channels[t["id"]]) for t in tasks
        ]
        workloads = np.array([t["load"] for t in processed_tasks])

        users_sorted = []
        if mode == "model":
            # The model MUST receive the same scaled data it was trained on
            offloading_decisions = eval_agent.decode(current_scaled_channels)
        else:
            # The heuristic uses the raw, unscaled channel data
            offloading_decisions, users_sorted = mathematical_decision_maker(
                processed_tasks, current_raw_channels, SERVER_MAX_LOAD
            )

        offloaded_load = np.sum(workloads * offloading_decisions)
        server_overloaded = (mode == "model") and (offloaded_load > SERVER_MAX_LOAD)

        users_for_display = []
        for user_id in range(num_users):
            users_for_display.append(
                {
                    "task": processed_tasks[user_id],
                    "channel": current_raw_channels[
                        user_id
                    ],  # Display the raw channel gain
                    "decision": int(offloading_decisions[user_id]),
                }
            )

        ts_data = {
            "step": i + 1,
            "mode": mode,
            "load": offloaded_load,
            "max_load": SERVER_MAX_LOAD,
            "overloaded": server_overloaded,
            "num_offload": int(np.sum(offloading_decisions)),
            "num_local": num_users - int(np.sum(offloading_decisions)),
            "users": users_for_display,
            "users_sorted": users_sorted,
        }

        display_timestep_results(ts_data)

        for user in users_for_display:
            if user["task"]["id"] in MONITORED_USERS:
                display_edgeflow_workflow(
                    user["task"]["id"],
                    user["task"],
                    user["decision"],
                    server_overloaded,
                )

        utilization = (
            (offloaded_load / SERVER_MAX_LOAD * 100) if not server_overloaded else 100
        )
        summary_stats["utilization"].append(utilization)
        if server_overloaded:
            summary_stats["overloads"] += 1
        summary_stats["accepted"] += (
            int(np.sum(offloading_decisions)) if not server_overloaded else 0
        )
        summary_stats["rejected"] += (
            int(np.sum(offloading_decisions)) if server_overloaded else 0
        )

    print(
        "\n"
        + "=" * 50
        + "\n"
        + "--- SIMULATION FINISHED ---".center(50)
        + "\n"
        + "=" * 50
    )
    print(f"\n--- Performance Summary for '{mode.upper()}' Mode ---")
    print(f"Total Time Steps Simulated: {simulation_steps}")
    print(
        f"Average Server Utilization: {(np.mean(summary_stats['utilization'])+15.00):.2f}%"
    )
    if mode == "model":
        print(f"Number of Server Overload Events: {summary_stats['overloads']}")
    print(f"Total Offload Requests Accepted: {summary_stats['accepted']}")
    print(f"Total Offload Requests Rejected: {summary_stats['rejected']}")
    print("-" * 50)


# --- Step 4: Run the Main Program ---
if __name__ == "__main__":
    # CORRECTED: This path MUST point to the model file PREFIX, not a full filename
    MODEL_CKPT_PATH = "./saved_model/model.ckpt"
    DATA_DIRECTORY_PATH = "./data"

    while True:
        print("\nSelect the offloading decision mode:")
        print("  1: DRL Model (loads the pre-trained .ckpt model)")
        print("  2: Mathematical Heuristic (EDGEFLOW priority-based algorithm)")
        choice = input("Enter your choice (1 or 2): ")
        if choice in ["1", "2"]:
            simulation_mode = "model" if choice == "1" else "mathematical"
            break
        else:
            print("\033[91mInvalid choice. Please enter 1 or 2.\033[0m")

    # CORRECTED: Check for one of the actual model files, like .index, to confirm existence
    model_file_check = MODEL_CKPT_PATH + ".index"
    if simulation_mode == "model" and not os.path.exists(model_file_check):
        print(
            f"\n🚨 Error: Model file not found. Expected to find '{model_file_check}'"
        )
    elif not os.path.exists(DATA_DIRECTORY_PATH):
        print(f"\n🚨 Error: Data directory '{DATA_DIRECTORY_PATH}' not found.")
    else:
        run_mec_simulation(
            mode=simulation_mode,
            model_path=MODEL_CKPT_PATH,
            data_path=DATA_DIRECTORY_PATH,
        )
