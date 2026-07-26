"""Agentic-SAM v2 — RFA surgical assistant. Entry point.

Run with: streamlit run app.py --server.port 7870
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Agentic-SAM v2 — RFA Assistant",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.navigation(
    [
        st.Page("ui/live_procedure.py", title="Live Procedure", icon="🫀", default=True),
        st.Page("ui/image_analysis.py", title="Image Analysis", icon="🖼️"),
        st.Page("ui/report.py", title="Procedure Report", icon="📄"),
        st.Page("ui/status.py", title="System Status", icon="🛠️"),
    ]
).run()
