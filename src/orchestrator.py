import json
from datetime import datetime
from pathlib import Path

from crewai import Crew, Process, Task

from src.agents.data_engineer import create_data_engineer
from src.agents.cleaning_agent import create_cleaning_agent
from src.agents.feature_engineer import create_feature_engineer
from src.agents.ml_engineer import create_ml_engineer
from src.agents.statistician import create_statistician
from src.agents.visualization_expert import create_visualization_expert
from src.agents.business_analyst import create_business_analyst
from src.config import get_settings, OUTPUTS_DIR, PROCESSED_DIR
from src.exceptions import CrewExecutionError
from src.logger import log
from src.models.schemas import ExperimentResult, ModelMetrics
from src.storage.experiment_store import ExperimentStore
from src.storage.dataset_versions import DatasetVersioner


class DataScienceOrchestrator:
    def __init__(self, dataset_path: str, run_name: str = ""):
        self.dataset_path = Path(dataset_path)
        self.run_name = run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.settings = get_settings()
        self.experiment_store = ExperimentStore()
        self.versioner = DatasetVersioner()
        self.output_dir = OUTPUTS_DIR / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_crew(self) -> Crew:
        data_engineer = create_data_engineer()
        cleaning_agent = create_cleaning_agent()
        feature_engineer = create_feature_engineer()
        statistician = create_statistician()
        ml_engineer = create_ml_engineer()
        viz_expert = create_visualization_expert()
        business_analyst = create_business_analyst()

        processed_dir = str(PROCESSED_DIR)

        task_data_load = Task(
            description=(
                f"Load the dataset from '{self.dataset_path}' using the data_loader tool, "
                f"then run the schema_inspector tool on it. Provide a comprehensive summary including: "
                f"row count, column count, data types for each column, missing value percentages, "
                f"basic statistics for numeric columns, and unique value counts for categorical columns. "
                f"The dataset info JSON should include: row_count, column_count, columns list, "
                f"total_missing_pct, and memory_usage_mb."
            ),
            expected_output=(
                "A detailed dataset profile JSON with: row/column counts, column names and types, "
                "missing value percentages, basic statistics, and data quality observations."
            ),
            agent=data_engineer,
        )

        task_clean = Task(
            description=(
                f"Clean the dataset at '{self.dataset_path}' using the cleaning tools:\n"
                f"1) Run missing_value_analyzer on '{self.dataset_path}' to analyze missing patterns\n"
                f"2) Run missing_value_imputer on '{self.dataset_path}' with strategy='auto' to impute missing values\n"
                f"3) Run outlier_detector on the cleaned file to detect outliers\n"
                f"4) The cleaned file will be saved to '{processed_dir}'\n"
                f"Provide a summary of all cleaning operations performed."
            ),
            expected_output=(
                "A cleaning summary detailing: missing values handled (method and count), "
                "outliers detected, and the path to the cleaned dataset in the processed directory."
            ),
            agent=cleaning_agent,
            context=[task_data_load],
        )

        task_features = Task(
            description=(
                f"Engineer features from the cleaned dataset:\n"
                f"1) Run feature_encoder on the cleaned CSV in '{processed_dir}' with method='auto'\n"
                f"2) Run feature_creator on the encoded file to create interaction features\n"
                f"3) Run feature_selector with target_column='Survived' to select top features\n"
                f"4) The feature-engineered file will be saved to '{processed_dir}'\n"
                f"Provide a summary of feature engineering operations."
            ),
            expected_output=(
                "A feature engineering summary detailing: encodings applied, new features created, "
                "features selected, total feature count, and the path to the feature-engineered dataset."
            ),
            agent=feature_engineer,
            context=[task_data_load, task_clean],
        )

        task_stats = Task(
            description=(
                f"Perform statistical analysis on the engineered features in '{processed_dir}':\n"
                f"1) Run descriptive_stats on the feature-engineered CSV\n"
                f"2) Run correlation_analyzer on the feature-engineered CSV\n"
                f"3) Run hypothesis_tester with column='Survived' and group_column='Sex' for t-test\n"
                f"Focus on relationships between features and the target variable.\n"
                f"Provide key statistical findings."
            ),
            expected_output=(
                "Statistical findings including: key distributions, significant correlations, "
                "hypothesis test results with p-values, and observations about feature relationships."
            ),
            agent=statistician,
            context=[task_data_load, task_features],
        )

        task_ml = Task(
            description=(
                f"Train and evaluate machine learning models on the engineered features:\n"
                f"1) Run model_trainer on the feature-engineered CSV in '{processed_dir}' "
                f"with target_column='Survived' and task_type='classification'\n"
                f"2) Run model_evaluator on the model results JSON in '{OUTPUTS_DIR}'\n"
                f"Provide a model comparison table with all metrics."
            ),
            expected_output=(
                "A model comparison table with: model names, accuracy, precision, recall, F1, "
                "AUC (if applicable), cross-validation scores, and identification of the best model."
            ),
            agent=ml_engineer,
            context=[task_data_load, task_features, task_stats],
        )

        task_viz = Task(
            description=(
                f"Generate visualizations for the analysis:\n"
                f"1) Run chart_generator on the feature-engineered CSV in '{processed_dir}' "
                f"with chart_types='distributions,heatmap,boxplot'\n"
                f"2) Run model_viz on the model results JSON in '{OUTPUTS_DIR}/model_results.json'\n"
                f"Focus on charts that would be most informative for stakeholders.\n"
                f"Charts will be saved to '{OUTPUTS_DIR}'"
            ),
            expected_output=(
                "A list of generated chart file paths and a brief description of what each chart shows."
            ),
            agent=viz_expert,
            context=[task_data_load, task_features, task_stats, task_ml],
        )

        task_report = Task(
            description=(
                f"Generate an executive summary report:\n"
                f"1) Run report_generator with the following inputs:\n"
                f"   - dataset_info: the JSON from the data loading task\n"
                f"   - cleaning_summary: the summary from the cleaning task\n"
                f"   - feature_summary: the summary from the feature engineering task\n"
                f"   - stats_summary: the findings from the statistical analysis task\n"
                f"   - model_results: the model comparison JSON from '{OUTPUTS_DIR}/model_results.json'\n"
                f"   - viz_charts: the chart paths JSON from the visualization task\n"
                f"2) The report will be saved to '{self.output_dir}/executive_report.md'"
            ),
            expected_output=(
                "A well-structured executive report in Markdown format with clear sections "
                "for each analysis phase, visualizations, and actionable recommendations."
            ),
            agent=business_analyst,
            context=[task_data_load, task_clean, task_features, task_stats, task_ml, task_viz],
            output_file=str(self.output_dir / "executive_report.md"),
        )

        crew = Crew(
            agents=[
                data_engineer,
                cleaning_agent,
                feature_engineer,
                statistician,
                ml_engineer,
                viz_expert,
                business_analyst,
            ],
            tasks=[
                task_data_load,
                task_clean,
                task_features,
                task_stats,
                task_ml,
                task_viz,
                task_report,
            ],
            process=Process.sequential,
            verbose=True,
            memory=self.settings.enable_memory,
            cache=True,
            max_rpm=self.settings.max_rpm,
            output_log_file=str(self.output_dir / "crew_logs.txt"),
            planning=False,
            planning_llm=self.settings.resolved_model,
        )

        return crew

    def run(self) -> dict:
        import time
        log.info("orchestrator_started", dataset=str(self.dataset_path), run_name=self.run_name)

        crew = self.build_crew()
        max_retries = 3
        retry_delay = 60

        for attempt in range(max_retries):
            try:
                result = crew.kickoff(inputs={"dataset_path": str(self.dataset_path)})
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "rate" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        log.warning("rate_limit_hit", retry=attempt + 1, wait_seconds=wait_time)
                        print(f"\nRate limit hit. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                log.error("crew_execution_failed", error=str(e))
                raise CrewExecutionError(f"Crew execution failed: {e}")

        log.info("orchestrator_completed", run_name=self.run_name)

        report_path = str(self.output_dir / "executive_report.md")
        experiment = ExperimentResult(
            dataset_hash="",
            run_name=self.run_name,
            models=[],
            best_model="",
            best_f1=0.0,
            feature_count=0,
            rows_used=0,
            report_path=report_path,
            status="completed",
        )

        model_results_path = OUTPUTS_DIR / "model_results.json"
        if model_results_path.exists():
            try:
                with open(model_results_path) as f:
                    data = json.load(f)
                results = data.get("results", [])
                valid = [r for r in results if "error" not in r]
                experiment.models = [ModelMetrics(**r) for r in valid]
                if valid:
                    best = max(valid, key=lambda x: x.get("f1", 0))
                    experiment.best_model = best.get("model_name", "")
                    experiment.best_f1 = best.get("f1", 0.0)
            except Exception:
                pass

        exp_id = self.experiment_store.log_experiment(experiment)

        return {
            "status": "completed",
            "run_name": self.run_name,
            "experiment_id": exp_id,
            "report_path": report_path,
            "output_dir": str(self.output_dir),
        }
