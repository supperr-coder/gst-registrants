"""
Streamlit frontend for the GST Entity Matcher.

Supports:
- Single entity lookup via text input
- Batch lookup via CSV upload (auto-detects the entity name column)
- Results displayed as an interactive table with score colour-coding
- CSV download of results
"""
import logging

import pandas as pd
import streamlit as st

from app.utils import parse_uploaded_csv, results_to_csv_bytes
from matching.pipeline import match_entities

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GST Entity Matcher",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 GST Entity Matcher")
st.caption(
    "Check whether company names are GST-registered in Singapore. "
    "Results are ranked by embedding similarity."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["Single Entity", "Batch CSV"])


# ── Single entity ────────────────────────────────────────────────────────────
with tab_single:
    entity_input = st.text_input(
        "Entity name",
        placeholder="e.g. ABC TRADING PTE LTD",
    )

    if st.button("Search", key="btn_single") and entity_input.strip():
        with st.spinner("Matching..."):
            results = match_entities([entity_input.strip()])

        if results["matched_entity"].isna().all():
            st.warning("No matches found above the similarity threshold.")
        else:
            st.dataframe(
                results.style.format({"score": "{:.4f}"}),
                use_container_width=True,
            )


# ── Batch CSV ─────────────────────────────────────────────────────────────────
with tab_batch:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    col_override = st.text_input(
        "Entity column name",
        placeholder="Leave blank to auto-detect",
        key="col_override",
    )

    if uploaded is not None:
        try:
            df, detected_col = parse_uploaded_csv(
                uploaded, entity_column=col_override.strip() or None
            )
            st.info(
                f"Using column **{detected_col}** "
                f"({len(df)} rows, {df[detected_col].notna().sum()} non-empty)"
            )

            if st.button("Run Matching", key="btn_batch"):
                query_names = (
                    df[detected_col].dropna().str.strip().unique().tolist()
                )
                with st.spinner(f"Matching {len(query_names)} unique entities..."):
                    results = match_entities(query_names)

                matched = results["matched_entity"].notna().sum()
                total = results["query_name"].nunique()
                st.success(
                    f"Done — {matched} result rows across {total} queries."
                )

                st.dataframe(
                    results.style.format({"score": "{:.4f}"}),
                    use_container_width=True,
                )

                st.download_button(
                    label="⬇ Download Results CSV",
                    data=results_to_csv_bytes(results),
                    file_name="gst_match_results.csv",
                    mime="text/csv",
                )

        except ValueError as exc:
            st.error(str(exc))
