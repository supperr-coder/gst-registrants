"""
Streamlit frontend for the GST Entity Matcher.

Deployed on Airbase. Calls the SageMaker endpoint on MAESTRO for matching —
no local FAISS, embedding API, or S3 access needed.

Supports:
- Single entity lookup via text input
- Batch lookup via CSV upload (auto-detects the entity name column)
- Results displayed as an interactive table
- CSV download of results
"""
import logging

import pandas as pd
import streamlit as st

from api_client import match_entities
from utils import parse_uploaded_csv, results_to_csv_bytes

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IRAS Entity Matcher",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 IRAS Entity Matcher")
st.caption(
    "Match provided entity name(s) to IRAS data warehourse name(s). "
    "Results are ranked by embedding similarity."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_one, tab_two, tab_three = st.tabs(["Company", "GST Company", "Individual"])


# ── Tab 1 ────────────────────────────────────────────────────────
with tab_one:
    entities_input = st.text_area(
        "Enter single or multiple entity names",
        placeholder="QWERTY PTE LTD\nABC SDN BHD\nXYZ TRADING",
        max_chars=2000,
        height="content"
    )  # if nothing is typed, entities_input = ""
    entities_input_list = [e for e in (entity.strip() for entity in entities_input.split("\n")) if e]
    search_clicked = st.button("Search", key="btn_single")

    if search_clicked or entities_input_list != []:  # Ctrl+Enter adds a newline
        if entities_input_list != []:
            with st.spinner("Matching..."):
                results = match_entities(entities_input_list)

            float_cols = results.select_dtypes('float').columns
            st.dataframe(
                results.style.format({col: "{:.4f}" for col in float_cols}, na_rep=""),
                use_container_width=True,
            )
            st.download_button(
                label="⬇ Download Results CSV",
                data=results_to_csv_bytes(results),
                file_name="gst_match_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("Please enter at least one entity name.")

# ── Tab 2 ─────────────────────────────────────────────────────────────
with tab_two:
    pass
    # uploaded = st.file_uploader("Upload CSV", type=["csv"])
    # col_override = st.text_input(
    #     "Entity column name",
    #     placeholder="Leave blank to auto-detect",
    #     key="col_override",
    # )

    # if uploaded is not None:
    #     try:
    #         df, detected_col = parse_uploaded_csv(
    #             uploaded, entity_column=col_override.strip() or None
    #         )
    #         st.info(
    #             f"Using column **{detected_col}** "
    #             f"({len(df)} rows, {df[detected_col].notna().sum()} non-empty)"
    #         )

    #         if st.button("Run Matching", key="btn_batch"):
    #             query_names = (
    #                 df[detected_col].dropna().str.strip().unique().tolist()
    #             )
    #             with st.spinner(f"Matching {len(query_names)} unique entities..."):
    #                 results = match_entities(query_names)

    #             matched = results["matched_entity"].notna().sum()
    #             total = results["query_name"].nunique()
    #             st.success(
    #                 f"Done — {matched} result rows across {total} queries."
    #             )

    #             st.dataframe(
    #                 results.style.format({"score": "{:.4f}"}),
    #                 use_container_width=True,
    #             )

    #             st.download_button(
    #                 label="⬇ Download Results CSV",
    #                 data=results_to_csv_bytes(results),
    #                 file_name="gst_match_results.csv",
    #                 mime="text/csv",
    #             )

    #     except ValueError as exc:
    #         st.error(str(exc))
