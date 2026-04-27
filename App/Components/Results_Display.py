"""
App/Components/Results_Display.py
Renders primary/differential diagnosis, confidence gauge, and clinical recommendations.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumothorax'
]

CLINICAL_RECS: dict[str, list[str]] = {
    "Pneumonia":          ["Antibiotic course (e.g. Azithromycin)", "Sputum culture", "Pulmonary follow-up in 48 h"],
    "Pneumothorax":       ["Urgent needle decompression assessment", "CT chest immediately", "Cardiothoracic consult"],
    "Cardiomegaly":       ["12-lead ECG", "Echocardiogram referral", "Restrict fluid intake"],
    "Effusion":           ["Thoracentesis evaluation", "Rule out cardiac failure", "Repeat imaging in 24 h"],
    "Consolidation":      ["Broad-spectrum antibiotics", "Oxygen therapy if SpO2 < 94%", "Review in 48 h"],
    "Edema":              ["Diuretic therapy", "Monitor urine output", "Cardiology consult"],
    "No Finding":         ["No acute intervention required", "Continue vitals monitoring", "Follow-up if symptoms persist"],
}
_DEFAULT_RECS = ["Specialist referral required", "Consolidate with lab results", "Continuous vital monitoring"]


class ResultDisplay:
    """Renders the full diagnosis panel."""

    classes = DISEASE_CLASSES

    def display(self, predictions: np.ndarray, confidence_threshold: float = 0.7) -> int:
        """
        Args:
            predictions: softmax probability array [NUM_CLASSES].
        Returns:
            top_idx: index of the top predicted class.
        """
        top_idx = int(predictions.argmax())
        top_conf = float(predictions[top_idx])
        top3_idx = predictions.argsort()[-3:][::-1]

        # ── Glassmorphism card wrapper ────────────────────────────────────────
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        col_gauge, col_text = st.columns([1, 1.5])

        with col_gauge:
            st.markdown("### 🧬 Confidence Score")
            gauge_color = (
                "#00c9a7" if top_conf > 0.75
                else "#ffc107" if top_conf > 0.5
                else "#ff5252"
            )
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(top_conf * 100, 1),
                number={"suffix": "%", "font": {"color": gauge_color, "size": 52}},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "gray"},
                    "bar": {"color": gauge_color},
                    "bgcolor": "rgba(255,255,255,0.04)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],  "color": "rgba(255,82,82,0.12)"},
                        {"range": [50, 75], "color": "rgba(255,193,7,0.12)"},
                        {"range": [75, 100],"color": "rgba(0,201,167,0.12)"},
                    ],
                },
            ))
            fig.update_layout(
                height=230,
                margin=dict(l=5, r=5, t=30, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Outfit"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_text:
            label = DISEASE_CLASSES[top_idx]
            badge = "✅" if top_conf > confidence_threshold else "⚠️"
            st.markdown(
                f"#### {badge} Primary Finding: "
                f"<span style='color:#4facfe;font-weight:700'>{label}</span>",
                unsafe_allow_html=True,
            )
            status_msg = (
                "**High confidence — suitable for clinical decision support.**"
                if top_conf > confidence_threshold
                else "**Moderate confidence — additional clinical corroboration recommended.**"
            )
            st.markdown(status_msg)
            st.markdown("---")
            st.markdown("**📊 Differential Diagnosis**")
            diff_cols = st.columns(3)
            for i, idx in enumerate(top3_idx):
                with diff_cols[i]:
                    st.metric(
                        label=DISEASE_CLASSES[idx],
                        value=f"{predictions[idx]:.1%}",
                    )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Clinical Recommendations ─────────────────────────────────────────
        st.markdown("### 📋 Clinical Recommendations")
        recs = CLINICAL_RECS.get(label, _DEFAULT_RECS)
        rec_cols = st.columns(len(recs))
        for col, rec in zip(rec_cols, recs):
            with col:
                st.info(rec)

        return top_idx
