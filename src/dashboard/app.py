import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import UPLOADS_DIR, OUTPUTS_DIR, get_settings
from src.orchestrator import DataScienceOrchestrator
from src.storage.experiment_store import ExperimentStore
from src.storage.dataset_versions import DatasetVersioner

st.set_page_config(page_title="AI Data Scientist", page_icon=":bar_chart:", layout="wide")

st.title("AI Data Scientist Platform")
st.markdown("*Upload a dataset and let our AI team analyze it end-to-end.*")

settings = get_settings()
store = ExperimentStore()
versioner = DatasetVersioner()

tab_upload, tab_run, tab_results, tab_experiments = st.tabs(["Upload", "Run Analysis", "Results", "Experiments"])

with tab_upload:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, JSON, or Parquet file",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help=f"Maximum file size: {settings.max_file_size_mb}MB",
    )

    if uploaded_file:
        file_path = UPLOADS_DIR / uploaded_file.name
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.success(f"File uploaded: {uploaded_file.name}")

        try:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif name.endswith(".json"):
                df = pd.read_json(file_path)
            elif name.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_excel(file_path)
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{len(df):,}")
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing %", f"{df.isnull().mean().mean() * 100:.1f}%")

            st.subheader("Data Preview")
            st.dataframe(df.head(20), width="stretch")

            st.session_state["dataset_path"] = str(file_path)
            st.session_state["dataset_name"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab_run:
    st.header("Run Analysis")

    if "dataset_path" not in st.session_state:
        st.info("Please upload a dataset first.")
    else:
        st.info(f"Dataset: {st.session_state.get('dataset_name', 'Unknown')}")
        run_name = st.text_input("Run Name", value=f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
        target_column = st.text_input("Target Column", value="Survived",
                                      help="Column to predict / select features against")
        group_column = st.text_input("Group Column", value="Sex",
                                     help="Grouping column for the statistician's t-test")

        if st.button("Start AI Analysis", type="primary", width="stretch"):
            with st.spinner("Running AI Data Scientist pipeline... This may take a few minutes."):
                try:
                    orchestrator = DataScienceOrchestrator(
                        dataset_path=st.session_state["dataset_path"],
                        run_name=run_name,
                        target_column=target_column,
                        group_column=group_column,
                    )
                    result = orchestrator.run()
                    st.session_state["last_result"] = result
                    st.success(f"Analysis completed! Experiment ID: {result['experiment_id']}")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

with tab_results:
    st.header("Results")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.success(f"Run: {result['run_name']} (ID: {result['experiment_id']})")

        report_path = Path(result["report_path"])
        if report_path.exists():
            st.subheader("Executive Report")
            with open(report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())

        output_dir = Path(result["output_dir"])
        chart_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.html"))
        if chart_files:
            st.subheader("Visualizations")
            for chart_file in chart_files:
                if chart_file.suffix == ".png":
                    st.image(str(chart_file), caption=chart_file.stem, width="stretch")
                elif chart_file.suffix == ".html":
                    st.components.v1.html(chart_file.read_text(), height=500)

        results_json = OUTPUTS_DIR / "model_results.json"
        if results_json.exists():
            st.subheader("Model Comparison")
            with open(results_json) as f:
                data = json.load(f)
            results = data.get("results", [])
            valid = [r for r in results if "error" not in r]
            if valid:
                st.dataframe(pd.DataFrame(valid), width="stretch")
    else:
        st.info("No results yet. Run an analysis first.")

with tab_experiments:
    st.header("Experiment History")

    experiments = store.list_experiments(limit=20)
    if experiments:
        exp_data = []
        for exp in experiments:
            exp_data.append({
                "ID": exp.id,
                "Run Name": exp.run_name,
                "Best Model": exp.best_model,
                "Best F1": exp.best_f1,
                "Features": exp.feature_count,
                "Status": exp.status,
                "Timestamp": exp.timestamp.strftime("%Y-%m-%d %H:%M"),
            })
        st.dataframe(pd.DataFrame(exp_data), width="stretch")

        selected_ids = st.multiselect(
            "Select experiments to compare",
            options=[exp.id for exp in experiments],
            format_func=lambda x: f"#{x} - {next(e.run_name for e in experiments if e.id == x)}",
        )
        if len(selected_ids) >= 2:
            comparison = store.compare_experiments(selected_ids)
            st.subheader("Comparison")
            st.dataframe(pd.DataFrame(comparison), width="stretch")
    else:
        st.info("No experiments yet. Run an analysis to see history.")
