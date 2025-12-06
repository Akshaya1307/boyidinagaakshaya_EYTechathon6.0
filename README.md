# 🚗 AutoOps+ — Agentic AI for Predictive Vehicle Maintenance

### *An EY Techathon Project by Akshaya (Akki)*

### *Built with Multi-Agent Architecture, Predictive Analytics, RCA/CAPA & UEBA Security*

---

## 🌟 Overview

AutoOps+ is an **Agentic AI–driven platform** that autonomously monitors vehicle health, predicts failures, engages customers, schedules service appointments, and generates manufacturing-level RCA/CAPA insights to continuously improve product quality.

This system uses a **Master Orchestrator Agent** that supervises multiple Worker Agents — each responsible for a core part of the automotive maintenance lifecycle.

> ⚡ **End result:** Higher vehicle uptime, better customer experience, reduced breakdowns, and actionable feedback for manufacturing teams.

---

## 🧠 System Architecture

### **Master Agent (Orchestrator)**

* Coordinates all worker agents
* Maintains workflow logic across diagnosis → engagement → scheduling → feedback
* Ensures security via UEBA monitoring
* Maintains full audit logs

---

## 👥 Worker Agents

### 1️⃣ **Data Analysis Agent**

* Processes real-time vehicle telematics & historical logs
* Uses **Isolation Forest** for anomaly detection
* Computes **RiskScore** (0–100) for failure likelihood

---

### 2️⃣ **Diagnosis Agent**

* Identifies highest-risk components (engine, battery, brake, cooling systems)
* Assigns priority levels: *Low / Medium / High / Critical*
* Generates reasoning based on sensor patterns

---

### 3️⃣ **Customer Engagement Agent**

* Simulated **voice-based agent** that explains predicted issues
* Chatbot interface for appointment intent
* Uses intent recognition to guide next steps

---

### 4️⃣ **Scheduling Agent**

* Checks service center capacity
* Books appointments based on customer preferences
* Prevents overbooking
* Logs all scheduling actions

---

### 5️⃣ **Feedback Agent**

* Collects customer satisfaction after service
* Records rating & comments
* Updates vehicle maintenance records

---

### 6️⃣ **Manufacturing Insights Agent (RCA/CAPA)**

Provides **professional-grade insights**:

* Identifies top anomaly-causing sensors
* Detects recurring failure patterns
* Generates dynamic RCA/CAPA recommendations
* Helps product teams improve future manufacturing batches

---

## 🔐 UEBA — User & Entity Behaviour Analytics

UEBA monitors all agent actions and detects:

* Abnormal orchestration behaviour
* Overbooking attempts
* Unauthorized agent activity
* High-severity operations
* Suspicious action keywords

If triggered, UEBA shows a **red security alert** and logs the event.

---

## 🖥️ Application UI (Streamlit)

The app is built using **Streamlit** with a **custom dark theme**.

Tabs include:

1. **Data & Monitoring** — Upload telematics, run master agent
2. **Diagnosis & Forecast** — Risk analysis + service demand forecasting
3. **Customer Voice Agent** — Voice script + chatbot
4. **Scheduling** — Center selection + capacity-aware booking
5. **Feedback & Manufacturing Insights** — RCA/CAPA engine
6. **UEBA Monitor** — Real-time agent security analytics

---

## 🛠️ Tech Stack

* **Python 3**
* **Streamlit** (Frontend + workflow UI)
* **scikit-learn** (IsolationForest model)
* **pandas / numpy**
* **Custom Agentic Architecture**
* **UEBA Rule-based Security Engine**

---

## 📁 File Structure

```
EY/
│── EY_final.py             → Main Streamlit app
│── engine_data.xlsx        → Sample dataset
│── requirements.txt        → Dependencies
└── .streamlit/
     └── config.toml        → Dark theme config
```

---

## 📊 Example Features Visible During Demo

* Anomaly detection & risk scoring
* Failure prediction (Engine, Brake, Battery, Cooling)
* Demand forecasting chart
* Voice agent script
* Scheduling with capacity enforcement
* Post-service feedback
* Advanced RCA/CAPA insights
* UEBA security alerts

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run EY_final.py
```

## 🌐 Deployment

The app can be deployed easily using **Streamlit Cloud**.

## 💡 Why AutoOps+ Stands Out

✔ Full end-to-end agentic workflow
✔ Real anomaly detection + RCA/CAPA
✔ Enterprise UEBA security
✔ Customer experience automation
✔ Manufacturing feedback loop
✔ Professional UI + Dark mode
✔ Scalable, production-aligned architecture

Naga Akshaya Boyidi
Techathon Participant • Future Data Scientist


Just say **“Vikky update README”**.
