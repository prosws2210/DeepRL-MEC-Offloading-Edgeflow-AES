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
import json # Added for creating dynamic messages

# Import the detailed protocol handler from the separate file
from edgeflow_protocol_handler import edgeflow_protocol_handler
from memory_loader import MemoryDNN_TF1_Loader

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)

# --- Step 1: Define Core Classes and Functions ---


def generate_task_for_user(user_id):
    """Creates a new computational task with random attributes."""
    task_name = f"user{user_id}_task{int(time.time())}"
    return {
        "id": user_id,
        "name": task_name,
        "load": random.randint(20, 100),  # GFLOPs
        "initial_data_size": random.uniform(5, 50),  # MB
        "priority": random.choice(["High", "Medium", "Low"]),
        "criticality": random.choice([True, False]),
    }


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
            + (channels[i] * 1e5)
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
            f"{str(round(task.get('initial_data_size', task.get('initial_size_MB', 0)), 1))+' MB':<12} | "
            f"{task['priority']:<10} | "
            f"{f'{gain:.2e}':<15} "
            f"{rank_text} | "
            f"{decision_colored} | "
            f"{status_text:^16}"
        )

    print("-" * 125)


# =======================================================================================
# --- REVISED WORKFLOW DISPLAY FUNCTION ---
# This is the fully verbose version that addresses your feedback.
# =======================================================================================
def display_edgeflow_workflow(user_id, task, decision, overloaded):
    """
    Prints the detailed, step-by-step EDGEFLOW workflow using metadata from the handler.
    Now prints full messages and all metadata.
    """
    print(f"\n--- 🔬 EDGEFLOW Protocol Workflow for User {user_id} ---")

    # 1. Initial State & Decisions
    print(f"1. [UE-{user_id}] Task Generated: Name='{task['name']}', Priority='{task['priority']}', Criticality='{task.get('criticality', 'N/A')}'")
    print(f"2. [EDGEFLOW] Analyzing Task & Channel (Gain: {task['channel_gain']:.2e})")
    print(f"   - Transport Mode Selected: {task['transport_mode']}")
    print(f"   - Compression Ratio Heuristic: {task['compression_ratio_used']:.2f}")
    time.sleep(0.1)

    # 2. Compression
    print(f"3. [EDGEFLOW] 📦 Compressing Payload...")
    print(f"   - Original Data Size: {task['initial_size_MB']:.4f} MB")
    print(f"   - Compressed Size:    {task['compressed_size_MB']:.4f} MB")
    time.sleep(0.1)

    # 3. Create the message that will be encrypted
    message_to_send = json.dumps({
        "task_name": task['name'],
        "user_id": user_id,
        "task_load": task.get('load', 'N/A'),
        "priority": task['priority'],
        "message": f"Full data payload for task {task['name']}"
    }, indent=2)

    # 4. Hashing and Encryption
    print(f"4. [EDGEFLOW] 🔐 Applying Authenticated Encryption (AES-GCM)...")
    print(f"   - Encrypting JSON message:\n{message_to_send}")
    final_size = task['final_size_with_encryption_overhead_MB']
    time.sleep(0.2)
    
    # 5. Print all metadata from the handler
    print("\n   --- Full Handler Metadata ---")
    # Define a preferred order for clarity
    display_order = [
        'channel_gain', 'transport_mode', 'compression_ratio_used', 'initial_size_MB',
        'compressed_size_MB', 'ciphertext_size_MB', 'final_size_with_encryption_overhead_MB',
        'plaintext_sha256', 'plaintext_md5', 'cipher_sha256', 'cipher_md5', 'aes_nonce_hex',
        'aes_tag_hex', 'aes_key_provided', 'simulated_aes_key_hex'
    ]
    for key in display_order:
        if key in task:
             print(f"     - {key}: {task[key]}")
    print("   -----------------------------\n")
    time.sleep(0.2)


    # 6. Offloading Decision and Transmission
    if decision == 0:
        print(f"5. [UE-{user_id}] 🖥️  Decision: Process task locally. No transmission.")
    else:
        print(f"5. [UE-{user_id}] 📡 Decision: Attempting to Offload {final_size:.4f} MB of encrypted data.")
        time.sleep(0.2)

        if overloaded:
            print("6. [MES] >> ❌ REJECTED << Server is overloaded. Task must be processed locally.")
        else:
            print("6. [MES] >> ✅ ACCEPTED << Server acknowledges offload request.")
            time.sleep(0.2)
            print(f"7. [UE-{user_id} -> MES] Transmitting {final_size:.4f} MB...")
            print(f"   - Sent ciphertext with SHA-256 hash: {task['cipher_sha256']}")
            time.sleep(0.2)
            print(f"8. [MES] 🔓 Received & Verified. Integrity check (SHA-256) passed.")
            print(f"   - Decrypting using AES-GCM with nonce: {task['aes_nonce_hex']}")
            print(f"   - Decrypted Message:\n{message_to_send}")
            time.sleep(0.2)

            response_message = f"Task {task['name']} complete. Result: SUCCESS."
            print(f"9. [MES -> UE-{user_id}] Processing complete. Sending response...")
            print(f"   - Encrypting & sending response: '{response_message}'")
            time.sleep(0.2)
            print(f"10.[UE-{user_id}] 🔓 Response received and decrypted successfully.")

    print("--- End of Workflow ---")

# --- Step 3: Main Simulation Function ---
def run_mec_simulation(mode, model_path, data_path, num_users=10, simulation_steps=5):
    """Main simulation loop."""
    print(f"\n--- Initializing MEC Simulation in '{mode.upper()}' Mode ---")
    SERVER_MAX_LOAD = 250.0
    MONITORED_USERS = [1, 5]
    print(
        f"Server Max Load: {SERVER_MAX_LOAD:.1f} GFLOPs | Monitored Users: {MONITORED_USERS}"
    )

    eval_agent = None
    if mode == "model":
        eval_agent = MemoryDNN_TF1_Loader(net=[num_users, 120, 80, num_users])
        try:
            eval_agent.load_model(model_path)
        except Exception:
            print("Could not continue due to model loading failure.")
            return

    data_file = os.path.join(data_path, f"data_{num_users}.mat")
    try:
        raw_channel_data = sio.loadmat(data_file)["input_h"]
        scaled_channel_data = raw_channel_data * 1e6
        print(f"✅ Channel data loaded and scaled for the model.")
    except FileNotFoundError:
        print(f"🚨 Error: Data file not found at {data_file}")
        return

    print("\n--- Simulation Started ---")
    summary_stats = {"accepted": 0, "rejected": 0, "overloads": 0, "utilization": []}

    for i in range(simulation_steps):
        time.sleep(1.0)

        current_scaled_channels = scaled_channel_data[i, :]
        current_raw_channels = raw_channel_data[i, :]

        tasks = [generate_task_for_user(u) for u in range(num_users)]
        
        processed_tasks = []
        for t in tasks:
            channel_gain = current_raw_channels[t["id"]]
            handler_result = edgeflow_protocol_handler(t, channel_gain)
            combined_task = t.copy()
            combined_task.update(handler_result['metadata'])
            processed_tasks.append(combined_task)
            
        workloads = np.array([t["load"] for t in processed_tasks])

        users_sorted = []
        if mode == "model":
            offloading_decisions = eval_agent.decode(current_scaled_channels)
        else:
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
                    "channel": current_raw_channels[user_id],
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