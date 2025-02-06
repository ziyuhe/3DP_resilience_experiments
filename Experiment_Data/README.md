# 📊 Experiment Results Overview  

This directory contains the results of **3DP resilience strategy experiments**, organized by study focus.

---

## **1️⃣ Basic Statistics on Synthetic Suppliers**  
📂 **Results:** [`Basic Statistics on Synthetic Products`](Experiment_Data/Basic_Pictures_Synthetic_Products/)  

- Analyzes **per-unit sales price, per-unit weight, and monthly demand** for synthetic suppliers.

---

## **2️⃣ Testing Computational Methods**  
📂 **Results:** [`MIP vs Supermodular Heuristics`](Experiment_Data/Compare_MIP_Time_GRB_Benders_BoE/)  

- Compares **Benders decomposition (MIP)** vs. **Supermodular Heuristics (BOE)** in terms of **runtime** and **optimality gaps**.

---

## **3️⃣ Initial Verification of 3DP's Impact**  
📂 **Results:** [`3DP vs. Dedicated Strategies`](Experiment_Data/Switch_Backup_vs_n/)  

- Evaluates **supplier backup shifts, cost savings, and demand shortfall reductions** under three strategies:  
  - **No 3DP** (Dedicated strategies only)  
  - **Naïve 3DP** (Only protects previously unprotected products)  
  - **Full 3DP** (Allows switching from dedicated strategies)  

---

## **4️⃣ Factors Driving the Switch to 3DP**  
📂 **Results:** [`Decision Tree Analysis`](Experiment_Data/Decision_Tree/)  

- Identifies key product characteristics influencing **switching to 3DP resilience**.  

---

## **5️⃣ Key Drivers of 3DP Resilience Benefits**  
We analyze **how key parameters affect 3DP cost savings & demand shortfall reduction**:

- 📂 [`Impact of 3DP Fixed Costs`](Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_3DPFixedCost/)  
- 📂 [`Impact of 3DP Variable Costs`](Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_c3DP/)  
- 📂 [`Impact of Disruption Rates & Yield Losses`](Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/)  
- 📂 [`Impact of Disruption Correlations`](Experiment_Data/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/)  
