import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils.llm_helper import call_llm
from utils.audit_logger import log

SYSTEM_PROMPT = """You are an enterprise cloud cost intelligence agent.

You will receive data about cloud infrastructure cost anomalies detected by ML.
For EACH anomaly:

1. DETECT: Confirm why this month is anomalous vs baseline
2. DIAGNOSE: Identify root cause:
   - Over Provisioning: Instances > 10 with usage < 90%
   - Traffic Spike: Usage > 90% with proportional instance growth
   - Misconfiguration: Cost spike without matching usage/instance growth
   - Auto-scaling Runaway: Instances grew faster than usage
3. RECOMMEND: Specific corrective action with exact numbers
4. REMEDIATION STEPS: Step-by-step fix with estimated savings

Format each anomaly as:
---ANOMALY: [Month]---
Observed: Cost=$[X], Usage=[Y]%, Instances=[Z]
Baseline Comparison: [comparison]
Root Cause: [ONE of the 4 causes]
Confidence: HIGH/MEDIUM/LOW
Recommended Action: [specific action with numbers]
Estimated Monthly Savings: $[amount]
Remediation Steps:
1. [immediate - within 24hrs]
2. [short term - within 1 week]
3. [long term - prevent recurrence]
Approval Required: YES/NO
"""


def anomaly_detection() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "cloud_cost.csv")
    df = pd.read_csv(csv_path)

    df['cost_per_instance'] = df['Cost'] / df['Instances']
    df['cost_per_usage'] = df['Cost'] / df['Usage']

    features = df[['Cost', 'Usage', 'Instances', 'cost_per_instance', 'cost_per_usage']].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = IsolationForest(contamination=0.25, random_state=42, n_estimators=100)
    df['anomaly_score'] = model.fit_predict(features_scaled)
    df['anomaly_confidence'] = model.score_samples(features_scaled)

    anomalies = df[df['anomaly_score'] == -1].copy()
    normal = df[df['anomaly_score'] == 1]

    baseline_avg_cost = normal['Cost'].mean() if len(normal) > 0 else df['Cost'].mean()
    baseline_avg_instances = normal['Instances'].mean() if len(normal) > 0 else df['Instances'].mean()

    llm_context = f"""
Baseline (normal months):
- Average Cost: ${baseline_avg_cost:,.0f}
- Average Instances: {baseline_avg_instances:.1f}

Detected Anomalies:
"""
    for _, row in anomalies.iterrows():
        cost_increase = ((row['Cost'] - baseline_avg_cost) / baseline_avg_cost * 100)
        llm_context += f"""
Month: {row['Month']}
- Cost: ${row['Cost']:,} ({cost_increase:+.1f}% vs baseline)
- Usage: {row['Usage']}%
- Instances: {row['Instances']} ({row['Instances'] - baseline_avg_instances:+.1f} vs baseline)
- Cost per Instance: ${row['cost_per_instance']:,.0f}
"""

    llm_analysis = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_message=llm_context,
        max_tokens=2000
    )

    total_anomaly_cost = anomalies['Cost'].sum()
    estimated_savings = total_anomaly_cost * 0.30

    log(
        agent="AnomalyAgent",
        action="cost_anomaly_detection",
        input_summary=f"Analyzed {len(df)} months of cloud cost data",
        output_summary=f"Detected {len(anomalies)} anomalous months: {list(anomalies['Month'])}. "
                      f"Potential savings: ${estimated_savings:,.0f}",
        reasoning="Isolation Forest with 5 engineered features. "
                 "LLM performed root cause analysis across 4 cause categories."
    )

    return {
        "raw_data": df,
        "anomalies": anomalies,
        "normal_baseline": normal,
        "llm_diagnosis": llm_analysis,
        "baseline_avg_cost": baseline_avg_cost,
        "estimated_savings": estimated_savings
    }
