## 📂 Data Preprocessing for Synthesized Products

The [`Problem_Data/Data_Preprocessing/`](Problem_Data/Data_Preprocessing/) directory outlines the steps for creating **synthesized products** based on **Bill-of-Lading (BOL) data**.

⚠ **Note:** The raw data was retrieved from a **third-party service** and is **not included** in this repository due to licensing restrictions. To use the scripts in [`Problem_Data/Data_Preprocessing/`](Problem_Data/Data_Preprocessing/), you must **provide your own dataset** with a similar structure.

---

### 📌 **Data Organization**
- Normally, each supplier would have a **separate folder** under [`Problem_Data/Data_Preprocessing/`](Problem_Data/Data_Preprocessing/), where the **demand data** for its **synthesized product** is stored.
- However, for **clarity and accessibility**, we provide:
  - 📂 **Monthly demand data** in [`Problem_Data/All_Suppliers_Monthly_Demand/`](Problem_Data/All_Suppliers_Monthly_Demand/)
  - 📂 **Weekly demand data** in [`Problem_Data/All_Suppliers_Weekly_Demand/`](Problem_Data/All_Suppliers_Weekly_Demand/)

---

### 🔍 **K-Means Demand Clustering**
- In our experiments, we apply **K-Means clustering** to the **monthly demand data** to reduce **scenarios** into **K representative demand levels**.
- The resulting **K-atom distributions** are then used as **input demand distributions** in our experiments.
- These processed demand distributions are available in [`Problem_Data/All/`](Problem_Data/All/), e.g.:

  📄 [`Mattel_All_Suppliers_Ave_Month_Weight_Quantity_3scenarios.csv`](Problem_Data/All/Mattel_All_Suppliers_Ave_Month_Weight_Quantity_3scenarios.csv)

⚠ **Important:** These are the **final processed files** used in our experiments.

