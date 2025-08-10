import streamlit as st
import numpy as np
import random
import time


def generate_task_for_user(user_id):
    return {
        "id": user_id,
        "load": random.randint(20, 100),  # GFLOPs
        "initial_data_size": random.uniform(5, 50),  # MB
        "priority": random.choice(["High", "Medium", "Low"]),
        "criticality": random.choice([True, False]),
    }


def edgeflow_protocol_handler(task, channel_gain):
    task["transport_mode"] = (
        "TCP-like (Reliable)" if task["criticality"] else "UDP-like (Fast)"
    )
    priority_map = {"High": 0.8, "Medium": 0.6, "Low": 0.4}
    compression_ratio = priority_map.get(task["priority"], 0.5)
    if channel_gain > 5e-5:
        compression_ratio *= 0.9
    task["compressed_size"] = task["initial_data_size"] * compression_ratio
    task["final_data_size"] = task["compressed_size"] + 0.5
    return task


def mathematical_decision_maker(tasks, channels, server_max_load):
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


def simulate_encryption(message):
    return message.encode("utf-8").hex()


def simulate_decryption(ciphertext):
    try:
        return bytes.fromhex(ciphertext).decode("utf-8")
    except:
        return "Decryption Failed"


def display_edgeflow_workflow(user_id, task, decision):
    st.subheader(f"📡 EDGEFLOW Protocol Workflow for User {user_id}")
    sample_message = "Hello, Good Morning"
    st.write(f"1. [UE-{user_id}] Task Initialized: Priority={task['priority']}, Size={task['initial_data_size']:.2f} MB")
    st.write(f"2. [EDGEFLOW] Applying Protocol: Mode={task['transport_mode']}, Final Size={task['final_data_size']:.2f} MB")

    if decision == 1:
        st.write(f"3. [UE-{user_id}] Attempting to offload task...")
        encrypted_message = simulate_encryption(sample_message)
        st.write(f"   - 🔐 Encrypted message: {encrypted_message}")
        st.write(f"4. [MES] ACCEPTED ✅ Task offloaded.")
        decrypted = simulate_decryption(encrypted_message)
        st.write(f"5. [MES] 🔓 Decrypted message: {decrypted}")
    else:
        st.write("3. Task processed locally.")


def main():
    st.title("📶 MEC Offloading Simulation using EDGEFLOW Heuristic")
    st.markdown("Simulate task offloading decisions for mobile edge computing.")

    num_users = st.slider("Number of Users", 5, 20, 10)
    steps = st.slider("Simulation Steps", 1, 50, 5)
    server_max_load = st.number_input("Server Max Load (GFLOPs)", value=250.0)
    monitored_users = st.multiselect("Users to Monitor", list(range(num_users)), default=[1, 5])

    if st.button("Run Simulation"):
        st.success("Simulation Started...")
        summary_stats = {"accepted": 0, "rejected": 0, "utilization": []}

        for step in range(steps):
            st.markdown(f"### 🔁 Step {step+1}")
            channels = np.random.uniform(1e-6, 1e-4, size=num_users)
            tasks = [generate_task_for_user(i) for i in range(num_users)]
            tasks = [edgeflow_protocol_handler(t, channels[t["id"]]) for t in tasks]
            decisions, sorted_users = mathematical_decision_maker(tasks, channels, server_max_load)
            load = sum(t["load"] * decisions[t["id"]] for t in tasks)
            overloaded = False

            st.write(f"🔋 Server Load: {load:.2f} / {server_max_load:.2f} GFLOPs")
            utilization = load / server_max_load * 100
            summary_stats["utilization"].append(utilization)

            accepted = int(sum(decisions))
            rejected = num_users - accepted
            summary_stats["accepted"] += accepted
            summary_stats["rejected"] += rejected

            st.write(f"📡 Offloaded: {accepted} | 🖥️ Local: {rejected}")
            st.progress(min(100, int(utilization)))

            for user in sorted_users:
                t = user["task"]
                if t["id"] in monitored_users:
                    display_edgeflow_workflow(t["id"], t, user["decision"])

        st.markdown("## 📊 Final Summary")
        st.write(f"Total Steps: {steps}")
        st.write(f"Average Server Utilization: {np.mean(summary_stats['utilization']):.2f}%")
        st.write(f"Total Offload Requests Accepted: {summary_stats['accepted']}")
        st.write(f"Total Rejected (processed locally): {summary_stats['rejected']}")


if __name__ == "__main__":
    main()
