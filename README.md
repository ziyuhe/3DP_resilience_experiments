# **3DP Resilience Strategies: Experimental Analysis**

This repository contains experiments that systematically evaluate the effectiveness of **3D printing (3DP) as a resilience strategy** in mitigating supplier disruptions. **3DP acts as a flexible backup** (one-machine-fits-all) but is constrained by limited capacity and higher variable costs. In contrast, **dedicated resilience strategies** provide unlimited capacity but require **product-specific fixed costs**, which scale linearly with system size, making them impractical for large supply chains.

## **Key Research Questions**
Our experiments aim to assess how **3DP can**:
- Protect previously unprotected suppliers.
- Streamline the use of dedicated resilience strategies.
- Reduce costs and mitigate demand shortfalls under supply and demand uncertainties.

---

## **Experiments Overview**
This repository contains several experiments designed to analyze different aspects of **3DP resilience strategies**.

### **1️⃣ Basic Statistics on Synthetic Suppliers**
📌 **Script:** [`Experiment_Basic_Plots_Mattel.m`](Experiment_Basic_Plots_Mattel.m)  
📊 **Results:** [`Experiment_Data/Basic_Pictures_Synthetic_Products/`](Experiment_Data/Basic_Pictures_Synthetic_Products/)  

- We process **bill-of-lading data from Mattel** to generate synthetic suppliers.
- We analyze **per-unit sales price, per-unit weight, and monthly demand** for each synthetic product.

---

### **2️⃣ Testing Computational Methods**
📌 **Script:** [`Experiments_Compare_MIP_BoE.m`](Experiments_Compare_MIP_BoE.m)  
📊 **Results:** [`Experiment_Data/Compare_MIP_Time_GRB_Benders_BoE/`](Experiment_Data/Compare_MIP_Time_GRB_Benders_BoE/)  

- We compare the scalability of **two computational approaches** for optimizing:
  - Supplier backup selection for **3DP**.
  - **3DP capacity** allocation.
  - **First-stage order quantity** from primary suppliers.
- We analyze **runtime** and **optimality gaps** as the number of suppliers increases.
- Methods tested:
  - **Benders Decomposition** (MIP formulation)
  - **Supermodular Heuristics** (BOE approximation)

---

### **3️⃣ Initial Verification of 3DP's Impact**
📌 **Script:** [`Experiments_Switch_Backup_Boxplots.m`](Experiments_Switch_Backup_Boxplots.m)  
📊 **Results:** [`Experiment_Data/Switch_Backup_vs_n/`](Experiment_Data/Switch_Backup_vs_n/)  

- We analyze how **3DP integration** changes current resilience practices based on dedicated strategies.
- Key metrics compared:
  - **Number of suppliers switching to 3DP backup.**
  - **Cost savings.**
  - **Reduction in demand shortfalls.**
- **Three strategies tested**:
  1. **No 3DP** → Dedicated strategies only.
  2. **Naïve 3DP** → 3DP only protects previously unprotected products.
  3. **Full 3DP** → 3DP is allowed to replace dedicated strategies.

---

### **4️⃣ Factors Driving the Switch to 3DP**
📌 **Script:** [`Experiments_Switch_Backup_DT.m`](Experiments_Switch_Backup_DT.m)  and [DT_Analysis.R](Experiment_Data/Decision_Tree/DT_Analysis.R)

📊 **Results:** [`Experiment_Data/Decision_Tree/`](Experiment_Data/Decision_Tree/)  

- We conduct **decision tree analysis** to identify factors driving the shift from dedicated strategies to **3DP resilience**.
- Features include **product characteristics** (e.g., profitability under 3DP production).
- Binary classification: **Switched to 3DP (Yes/No).**

---

### **5️⃣ Key Drivers of 3DP Resilience Benefits**
📌 **Scripts:** Multiple (`Experiments_CostSavings_and_DemandShortfalls_*.m`)  
📊 **Results:** [`Experiment_Data`](Experiment_Data/)  

We systematically investigate how key parameters impact **3DP’s performance** in cost savings and demand shortfall reduction.

#### **Factors Explored:**
- **Fixed cost of 3D printers**  
  - 📂 [`Experiment_Data/3DPFixedCost/`](Experiment_Data/3DPFixedCost/)  
- **Variable cost of 3DP**  
  - 📂 [`Experiment_Data/c3DP/`](Experiment_Data/c3DP/)  
- **Disruption modeling** (independent disruptions with varying failure & yield loss rates)  
  - 📂 [`Experiment_Data/p_yieldloss/`](Experiment_Data/p_yieldloss/)  
- **Correlation among disruptions** (modeled using an interpolation framework)  
  - 📂 [`Experiment_Data/Corr_Interpolate/`](Experiment_Data/Corr_Interpolate/)  

---

## **📌 How to Use This Repository**
### **🔧 Requirements**
- MATLAB (Tested on R2023a+)
- Gurobi (For MIP-based optimization)
- Decision Tree & Machine Learning Toolbox (For decision tree analysis)

### **📂 Repository Structure**
