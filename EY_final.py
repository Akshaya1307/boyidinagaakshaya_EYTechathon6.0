import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import random

st.set_page_config(
    page_title="AutoOps+ – Agentic AI for Vehicle Maintenance",
    layout="wide"
)

# ---------------------------
# Session State Initialisation
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "anomaly_df" not in st.session_state:
    st.session_state.anomaly_df = None

if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None

if "appointments" not in st.session_state:
    st.session_state.appointments = []

if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# ---------------------------
# UEBA / Logging Helpers
# ---------------------------
def log_action(agent, action, details=None, severity="NORMAL"):
    log_entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent": agent,
        "action": action,
        "details": details,
        "severity": severity
    }
    st.session_state.logs.append(log_entry)
    return check_ueba_alert(log_entry)

def check_ueba_alert(last_log):
    """
    Simple UEBA rules:
    - If severity is HIGH → raise alert
    - If SchedulingAgent overbooks a center → alert
    - If too many actions from same agent within short period (simulated)
    """
    alert = None

    if last_log["severity"] == "HIGH":
        alert = f"UEBA Alert: High-severity action by {last_log['agent']} – {last_log['action']}"

    # Simple heuristic: suspicious action keywords
    suspicious_keywords = ["override", "force", "bypass", "overbook", "unauthorized"]
    if any(k in (last_log["action"] or "").lower() for k in suspicious_keywords):
        alert = f"UEBA Alert: Suspicious behaviour detected in {last_log['agent']} – {last_log['action']}"

    return alert


# ---------------------------
# Worker Agents
# ---------------------------

def data_analysis_agent(df, contamination=0.05, n_estimators=100):
    """Uses Isolation Forest to detect anomalous vehicle states."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        st.warning("No numeric columns found for anomaly detection.")
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42
    )
    preds = model.fit_predict(scaled)
    scores = model.decision_function(scaled)

    result = df.copy()
    result["Anomaly"] = np.where(preds == -1, "Anomaly", "Normal")
    # Lower score = more anomalous → convert to risk %
    risk_scores = -scores
    risk_norm = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-9)
    result["RiskScore"] = (risk_norm * 100).round(1)

    return result


def map_record_to_component(record):
    """
    Infer which component is at risk based on column names.
    This is generic so it works with your Kaggle telematics CSV.
    """
    col_names = [c.lower() for c in record.index]

    engine_keywords = ["engine", "rpm", "coolant", "temp", "load"]
    brake_keywords = ["brake", "brk"]
    battery_keywords = ["battery", "volt", "soc"]
    cooling_keywords = ["cooling", "fan", "radiator"]

    def has_keywords(keywords):
        return any(any(k in c for k in keywords) for c in col_names)

    if has_keywords(engine_keywords):
        return "Engine System"
    elif has_keywords(brake_keywords):
        return "Brake System"
    elif has_keywords(battery_keywords):
        return "Battery / Electrical"
    elif has_keywords(cooling_keywords):
        return "Cooling System"
    else:
        return "General Powertrain"


def diagnosis_agent(anomaly_df):
    """
    Picks the highest-risk anomalous vehicle and assigns
    a probable component failure + priority.
    """
    anomalies = anomaly_df[anomaly_df["Anomaly"] == "Anomaly"]
    if anomalies.empty:
        return {
            "component": "None",
            "risk": 0,
            "priority": "Low",
            "reason": "No anomalies detected.",
            "record_index": None
        }

    # Select the record with highest risk
    top = anomalies.sort_values("RiskScore", ascending=False).iloc[0]
    component = map_record_to_component(top)
    risk = float(top["RiskScore"])

    if risk > 85:
        priority = "Critical"
    elif risk > 70:
        priority = "High"
    elif risk > 50:
        priority = "Medium"
    else:
        priority = "Low"

    reason = f"Detected anomalous telemetry pattern with risk score {risk:.1f} for {component}."

    return {
        "component": component,
        "risk": risk,
        "priority": priority,
        "reason": reason,
        "record_index": top.name
    }


def demand_forecast_agent(anomaly_df):
    """
    Very simple 'forecast': estimate likely daily service demand based on
    count of anomalies per day or per batch index.
    """
    if anomaly_df is None:
        return None

    df = anomaly_df.copy()
    # If there is a time column, try to use it; else create pseudo-date buckets
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if time_cols:
        df["__date"] = pd.to_datetime(df[time_cols[0]], errors="coerce").dt.date
    else:
        # Fake dates based on index to still show a chart
        df["__date"] = pd.to_datetime("2025-12-01") + pd.to_timedelta(df.index, unit="D")

    demand = (
        df.groupby("__date")["Anomaly"]
        .apply(lambda x: (x == "Anomaly").sum())
        .reset_index(name="ExpectedHighRiskServices")
    )
    return demand


def scheduling_agent(preferred_date, preferred_slot, center_choice):
    """
    Very simple in-memory scheduling agent with capacity.
    """
    # Dummy capacities
    center_capacity = {
        "City Service Center A": 5,
        "City Service Center B": 3,
        "Premium Workshop": 2
    }

    if center_choice not in center_capacity:
        center_capacity[center_choice] = 5

    # Count existing appointments for that center & date
    existing = [
        a for a in st.session_state.appointments
        if a["center"] == center_choice and a["date"] == preferred_date
    ]
    if len(existing) >= center_capacity[center_choice]:
        alert = log_action(
            "SchedulingAgent",
            f"Overbook attempt for {center_choice} on {preferred_date}",
            severity="HIGH"
        )
        return None, alert

    appt = {
        "id": len(st.session_state.appointments) + 1,
        "center": center_choice,
        "date": preferred_date,
        "slot": preferred_slot,
        "status": "Booked"
    }
    st.session_state.appointments.append(appt)
    alert = log_action("SchedulingAgent", f"Booked appointment #{appt['id']} at {center_choice}")
    return appt, alert


def manufacturing_insights_agent(anomaly_df, diagnosis):
    """
    Generate simple RCA/CAPA-style insights based on anomalies & components.
    """
    if anomaly_df is None or diagnosis is None:
        return "Not enough data to generate insights."

    total = len(anomaly_df)
    anomalies = (anomaly_df["Anomaly"] == "Anomaly").sum()
    rate = (anomalies / total * 100) if total else 0

    component = diagnosis["component"]
    priority = diagnosis["priority"]

    insight_lines = [
        f"- Overall anomaly rate across fleet: **{rate:.1f}%**",
        f"- Highest risk observed in: **{component}** (Priority: **{priority}**)",
    ]

    # Simple CAPA suggestion based on component
    if "Engine" in component:
        capa = "Recommend firmware update for engine control unit and review cooling system margins in current design."
    elif "Brake" in component:
        capa = "Recommend material review for brake pads and calibration check for brake pressure sensors."
    elif "Battery" in component:
        capa = "Suggest revisiting battery thermal management design and charging profile in manufacturing guidelines."
    else:
        capa = "Recommend cross-functional design review for powertrain components and update preventive maintenance schedule."

    insight_lines.append(f"- CAPA Suggestion: {capa}")
    return "\n".join(insight_lines)


# ---------------------------
# Master Agent (Orchestrator)
# ---------------------------

def master_agent_pipeline(df, contamination, n_estimators):
    log_action("MasterAgent", "Starting health analysis pipeline")
    anomaly_df = data_analysis_agent(df, contamination, n_estimators)
    if anomaly_df is None:
        return None, None
    diagnosis = diagnosis_agent(anomaly_df)
    log_action("DiagnosisAgent", f"Diagnosis: {diagnosis['component']} at {diagnosis['risk']:.1f}% risk")
    st.session_state.anomaly_df = anomaly_df
    st.session_state.diagnosis = diagnosis
    return anomaly_df, diagnosis


# ---------------------------
# UI LAYOUT
# ---------------------------

st.title("🚗 AutoOps+ – Agentic AI for Predictive Vehicle Maintenance")

st.caption(
    "Master Agent orchestrating Data Analysis, Diagnosis, Customer Engagement, Scheduling, "
    "Feedback, Manufacturing Insights & UEBA."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Data & Monitoring",
    "2️⃣ Diagnosis & Forecast",
    "3️⃣ Customer Voice Agent",
    "4️⃣ Scheduling",
    "5️⃣ Feedback & Manufacturing Insights",
    "6️⃣ UEBA Monitor"
])

# ---------------------------
# TAB 1 – Data & Monitoring
# ---------------------------
with tab1:
    st.subheader("Vehicle Telematics & Continuous Monitoring")

    uploaded = st.file_uploader("Upload vehicle telematics CSV from Kaggle", type=["csv"])
    contamination = st.slider("Anomaly contamination rate", 0.01, 0.2, 0.05, 0.01)
    n_estimators = st.slider("Isolation Forest trees", 50, 300, 100, 10)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.write("Sample of uploaded data:")
        st.dataframe(df.head())

        if st.button("Run Master Agent – Analyze Vehicle Health"):
            anomaly_df, diagnosis = master_agent_pipeline(df, contamination, n_estimators)
            if anomaly_df is not None:
                st.success("Master Agent: Analysis completed.")
                st.write("Anomalies detected (if any):")
                st.dataframe(anomaly_df.head())
    else:
        st.info("Please upload a telematics CSV file to start monitoring.")

# ---------------------------
# TAB 2 – Diagnosis & Forecast
# ---------------------------
with tab2:
    st.subheader("Predictive Diagnosis & Service Demand Forecast")

    if st.session_state.anomaly_df is None or st.session_state.diagnosis is None:
        st.info("Run the Master Agent in Tab 1 to see diagnosis and forecast.")
    else:
        diag = st.session_state.diagnosis

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Diagnosis Agent Output")
            st.metric("Component at Risk", diag["component"])
            st.metric("Failure Risk (%)", f"{diag['risk']:.1f}")
            st.metric("Priority Level", diag["priority"])
            st.write("**Reasoning:**")
            st.write(diag["reason"])

        with col2:
            st.markdown("### Service Demand Forecast (High-Risk Vehicles)")
            demand = demand_forecast_agent(st.session_state.anomaly_df)
            if demand is not None and not demand.empty:
                st.line_chart(demand.set_index("__date")["ExpectedHighRiskServices"])
                st.write(demand)
            else:
                st.info("Not enough data to build a demand forecast.")

# ---------------------------
# TAB 3 – Customer Voice Agent (simulated)
# ---------------------------
with tab3:
    st.subheader("Voice-based Customer Engagement (Simulated)")

    if st.session_state.diagnosis is None:
        st.info("Run diagnosis first (Tabs 1 & 2) to personalize this conversation.")
    else:
        diag = st.session_state.diagnosis
        st.write(
            "Voice Agent speaking to the customer. "
        )

        sample_script = f"""
        **Voice Agent**

        > Hello, this is AutoOps+ calling about your vehicle.
        > Our predictive diagnostics system has detected an increased risk in your **{diag['component']}**.
        > The current failure risk is estimated at **{diag['risk']:.1f}%**, marked as **{diag['priority']} priority**.
        > To keep your vehicle safe and avoid unexpected breakdowns, we recommend a preventive service.
        > Would you like to book an appointment at your preferred service center and time?
        """

        st.markdown(sample_script)

        st.markdown("---")
        st.markdown("### Chatbot Simulation (Text)")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            role = "👩‍💻 You" if msg["role"] == "user" else "🤖 Agent"
            st.markdown(f"**{role}:** {msg['content']}")

        user_msg = st.text_input("Type what the customer would say (e.g., 'Yes, book for tomorrow morning'):")

        if st.button("Send Message") and user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            # Simple intent logic
            reply = ""
            text = user_msg.lower()
            if any(w in text for w in ["yes", "ok", "book", "sure"]):
                reply = "Great! Please proceed to the Scheduling tab to confirm your preferred center and time. I’ll pre-fill the risk details."
            elif any(w in text for w in ["no", "not now", "later"]):
                reply = "No problem. I’ll send you a reminder notification in a few days, and you can also schedule anytime from the mobile app."
            elif "issue" in text or "problem" in text:
                reply = "I understand your concern. The issue is only a predicted risk at this stage – there is no active failure yet, but early service can prevent future breakdowns."
            else:
                reply = "Thank you for your response. I recommend scheduling a preventive checkup soon to avoid unplanned downtime."

            st.session_state.chat_history.append({"role": "agent", "content": reply})
            log_action("CustomerEngagementAgent", f"Handled message: {user_msg[:30]}...")

            st.rerun()

# ---------------------------
# TAB 4 – Scheduling
# ---------------------------
with tab4:
    st.subheader("Autonomous Scheduling Agent")

    if st.session_state.diagnosis is None:
        st.info("Run diagnosis (Tabs 1 & 2) and engage the customer (Tab 3) before scheduling.")
    else:
        centers = ["City Service Center A", "City Service Center B", "Premium Workshop"]
        center_choice = st.selectbox("Select Service Center", centers)
        preferred_date = st.date_input("Preferred Date")
        preferred_slot = st.selectbox(
            "Preferred Time Slot",
            ["09:00–11:00", "11:00–13:00", "14:00–16:00", "16:00–18:00"]
        )

        if st.button("Confirm Appointment"):
            appt, alert = scheduling_agent(preferred_date, preferred_slot, center_choice)
            if appt:
                st.success(
                    f"Appointment #{appt['id']} booked at {appt['center']} on {appt['date']} ({appt['slot']})."
                )
            if alert:
                st.error(alert)

        if st.session_state.appointments:
            st.markdown("### Current Appointments")
            st.dataframe(pd.DataFrame(st.session_state.appointments))

# ---------------------------
# TAB 5 – Feedback & Manufacturing Insights
# ---------------------------
with tab5:
    st.subheader("Service Feedback & Manufacturing Quality Insights")

    if st.session_state.appointments:
        st.markdown("### Post-Service Feedback (Feedback Agent)")
        last_appt = st.session_state.appointments[-1]
        st.write(f"Latest appointment: #{last_appt['id']} at {last_appt['center']} on {last_appt['date']}")

        rating = st.slider("Customer Satisfaction Rating", 1, 5, 5)
        comments = st.text_area("Comments")

        if st.button("Submit Feedback"):
            feedback = {
                "appointment_id": last_appt["id"],
                "rating": rating,
                "comments": comments,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.feedbacks.append(feedback)
            log_action("FeedbackAgent", f"Captured feedback for appt #{last_appt['id']}")
            st.success("Feedback recorded successfully.")

    if st.session_state.feedbacks:
        st.markdown("### Feedback History")
        st.dataframe(pd.DataFrame(st.session_state.feedbacks))

def manufacturing_insights_agent(anomaly_df, diagnosis):
    """
    Professional RCA/CAPA engine:
    - Clusters anomalies by sensor groups
    - Identifies recurring failure patterns
    - Generates component-level heatmap insights
    - Produces automated CAPA recommendations
    """

    if anomaly_df is None or diagnosis is None:
        return "Insufficient data to generate RCA/CAPA insights."

    # 1. Basic stats
    total_records = len(anomaly_df)
    total_anomalies = (anomaly_df["Anomaly"] == "Anomaly").sum()
    anomaly_rate = (total_anomalies / total_records * 100) if total_records else 0

    # 2. Identify top anomaly-causing columns
    numeric_cols = anomaly_df.select_dtypes(include=[np.number]).columns

    anomaly_subset = anomaly_df[anomaly_df["Anomaly"] == "Anomaly"]
    if anomaly_subset.empty:
        return "No anomalies detected. No manufacturing RCA required."

    # Compute anomaly contribution by feature
    feature_contributions = {}
    for col in numeric_cols:
        try:
            feature_contributions[col] = anomaly_subset[col].std()
        except:
            continue

    # Sort by highest variability (indicating unstable behavior)
    sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)

    top_features = [f for f, _ in sorted_features[:5]]

    # 3. Component mapping
    component = diagnosis["component"]

    # 4. Frequent pattern detection
    pattern_summary = []
    for col in top_features:
        if anomaly_subset[col].mean() > anomaly_df[col].mean():
            pattern_summary.append(f"- {col}: consistently **above normal**, indicating abnormal stress")
        else:
            pattern_summary.append(f"- {col}: fluctuating patterns indicating **instability**")

    # 5. CAPA recommendations (dynamic)
    capa_suggestions = {
        "Engine": "Review engine thermal design thresholds, recalibrate coolant temperature sensors, and verify ECU mappings.",
        "Brake": "Perform friction material review, update brake pad material specifications, and recalibrate hydraulic sensors.",
        "Battery": "Update BMS firmware to adjust charging thresholds and investigate cell pack thermal imbalances.",
        "Cooling": "Inspect radiator airflow design, cooling fan algorithms, and coolant pump duty cycles."
    }

    # Match CAPA to component
    if "Engine" in component:
        capa = capa_suggestions["Engine"]
    elif "Brake" in component:
        capa = capa_suggestions["Brake"]
    elif "Battery" in component:
        capa = capa_suggestions["Battery"]
    elif "Cooling" in component:
        capa = capa_suggestions["Cooling"]
    else:
        capa = "Perform cross-system diagnostic review and update component preventive maintenance schedules."

    # 6. Prepare final insight block
    insights = f"""
### 🔧 Manufacturing RCA/CAPA Insights

**Overall Anomaly Rate:** {anomaly_rate:.1f}% of vehicles show abnormal sensor patterns  
**Highest Risk Component:** **{component}**  
**Detected Risk Level:** {diagnosis['risk']:.1f}% ({diagnosis['priority']})  

---

### 🧩 Recurring Failure Patterns Identified

Top anomaly-contributing sensors:
{chr(10).join([f"- **{f}**" for f in top_features])}

Behavioral patterns:
{chr(10).join(pattern_summary)}

---

### 🏭 Recommended CAPA Actions

{capa}

---

### 🚀 Impact

Implementing these corrective measures can:
- Reduce repeat failures  
- Improve component durability  
- Lower warranty claims  
- Feed improvements back to the design & manufacturing teams  
    """

    log_action("ManufacturingInsightsAgent", "Generated advanced data-driven RCA/CAPA insights")
    return insights


# ---------------------------
# TAB 6 – UEBA Monitor
# ---------------------------
with tab6:
    st.subheader("UEBA – User & Entity Behaviour Analytics for Agents")

    st.write(
        "This panel monitors actions performed by autonomous agents (Diagnosis, Scheduling, Customer Engagement, etc.) "
        "and flags suspicious behaviour."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simulate Abnormal Agent Behaviour"):
            alert = log_action(
                "SchedulingAgent",
                "Force override to book beyond capacity (simulated)",
                severity="HIGH"
            )
            if alert:
                st.error(alert)

    with col2:
        st.info("Use this button in your demo to show UEBA blocking abnormal orchestration actions.")

    if st.session_state.logs:
        st.markdown("### Recent Agent Actions")
        logs_df = pd.DataFrame(st.session_state.logs)
        st.dataframe(logs_df.tail(20))
    else:
        st.info("No agent actions logged yet.")
