# Deep Reinforcement Learning for Online Offloading in MEC Networks

## 📌 Project Overview
This project implements **Deep Reinforcement Learning-based Online Offloading (DROO)** for Mobile-Edge Computing (MEC) networks.  
It addresses the complex problem of **real-time binary offloading** in wireless powered MEC systems with multiple users competing for shared resources.  
By leveraging DRL, the system achieves **low latency**, **energy efficiency**, and **near-optimal performance** without relying on computationally expensive optimization methods.

---

## 🚀 Key Features
- **DRL-based Decision Making:** Real-time computation offloading decisions for multiple wireless devices.
- **Wireless Powered MEC Model:** Joint energy harvesting and task offloading.
- **Binary Offloading Policy:** Local computation vs. MEC server execution.
- **Memory-Augmented DNN:** Efficient learning from past experiences.
- **Robustness & Adaptability:** Handles dynamic scenarios like changing user priorities and fluctuating device availability.
- **Performance Benchmarking:** Compared against a theoretical oracle optimizer.

---

## 🛠️ System Architecture
- **State:** Vector of current wireless channel gains for all users.
- **Action:** Binary vector indicating offloading or local processing for each device.
- **Reward:** Weighted sum computation rate (performance metric to maximize).

---

## 📊 Methodology
1. **Environment Setup**
   - Wireless Powered MEC system simulation
   - Dynamic channel gain fluctuations
   - Multi-user resource contention
2. **DRL Agent Training**
   - Experience replay memory
   - Epsilon-greedy exploration
   - Adam optimizer with tuned hyperparameters
3. **Performance Evaluation**
   - Comparison with oracle optimization
   - Generalization to unseen scenarios
4. **Dynamic Scenario Testing**
   - Changing user weights
   - Users joining/leaving network

---

## 📈 Results Summary
- **Decision latency:** Reduced from seconds to <0.1s for 30 users.
- **Performance:** Achieves near-oracle weighted sum computation rates.
- **Adaptability:** Maintains efficiency under dynamic network conditions.

---

## 🔮 Future Enhancements
- Integration of **multi-agent reinforcement learning** for decentralized control.
- Incorporation of **network security protocols** (SHA/MD5 encryption, data compression).
- Real-world deployment testing on MEC-enabled IoT networks.

---

## 📚 References
- DROO: Deep Reinforcement Learning-based Online Offloading in Wireless Powered MEC Networks.
- Mobile-Edge Computing: A Key Technology Towards 5G.
