# 📦 Synthesized Product Creation from BOL Data  

The scripts in this directory are designed to **create synthesized products** based on **Bill-of-Lading (BOL) data**.  

⚠️ **Note:** The original **BOL data is retrieved from a third-party service** and is **not included** in this repository due to **licensing restrictions**. To use these scripts, you must provide your own dataset with a **similar structure**.  

---

## 🛠️ Steps for Synthesized Product Creation  

### 🔍 **1. Keyword Matching & Product Categorization**  
- **[`Mattel_Suppliers_Synthetic_Products_Info.py`](Mattel_Suppliers_Synthetic_Products_Info.py)**  
  - Matches **keywords** across different product categories in BOL raw descriptions.  
  - Uses keyword matching records to **estimate the proportion mix** of product categories in a supplier's BOL data.  
  - *(See `categories_keywords` inside the script for an example keyword list.)*  

### 📊 **2. Cost & Price Estimation**  
- **[`Mattel_Suppliers_Synthetic_Products_Info_Costs.py`](Mattel_Suppliers_Synthetic_Products_Info_Costs.py)**  
  - Computes **sales price** and **cost parameters** for the synthesized product.  
  - Uses **weighted averages** based on the proportion mix estimated in Step 1.  

### 📦 **3. Sourcing Data Processing**  
- **[`Mattel_Suppliers_Shipments_Processing.py`](Mattel_Suppliers_Shipments_Processing.py)**  
  - Converts **weight-based sourcing records** into **the number of units** of synthesized products per supplier.  

- **[`Mattel_Suppliers_Shipments_Processing_Aggregate_by_Month_Week.py`](Mattel_Suppliers_Shipments_Processing_Aggregate_by_Month_Week.py)**  
  - Aggregates **monthly and weekly sourcing quantities** for each supplier.  

### 📈 **4. Demand & Shipment Analysis**  
- **[`Mattel_Suppliers_Shipment_Kmeans.py`](Mattel_Suppliers_Shipment_Kmeans.py)**  
  - Applies **K-means clustering** on **monthly sourcing quantities** of synthesized products.  

- **[`Mattel_Suppliers_Shipment_Histogram.py`](Mattel_Suppliers_Shipment_Histogram.py)**  
  - Generates **histograms** of sourcing quantities per supplier.  

- **[`Mattel_Crude_Total_Month_Demand_in_weight.py`](Mattel_Crude_Total_Month_Demand_in_weight.py)**  
  - Computes **total monthly weight and quantity sourced** across all suppliers.  

