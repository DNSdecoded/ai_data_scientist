import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_settings
from src.logger import log


def main():
    parser = argparse.ArgumentParser(description="AI Data Scientist Platform")
    parser.add_argument("--file", "-f", type=str, help="Path to dataset file (CSV, Excel, JSON, Parquet)")
    parser.add_argument("--dashboard", "-d", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--model", "-m", type=str, help="LLM model identifier (e.g., gemini/gemini-3.5-flash)")
    parser.add_argument("--run-name", "-r", type=str, default="", help="Name for this analysis run")
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["LLM_MODEL"] = args.model

    if args.dashboard:
        import subprocess
        log.info("launching_dashboard")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
                        "--server.port", "8501", "--server.headless", "true"])
        return

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        log.info("pipeline_started", file=str(file_path))
        from src.orchestrator import DataScienceOrchestrator

        orchestrator = DataScienceOrchestrator(
            dataset_path=str(file_path),
            run_name=args.run_name,
        )
        result = orchestrator.run()

        print("\n" + "=" * 60)
        print("  AI Data Scientist - Analysis Complete")
        print("=" * 60)
        print(f"  Run Name:    {result['run_name']}")
        print(f"  Experiment:  #{result['experiment_id']}")
        print(f"  Report:      {result['report_path']}")
        print(f"  Output Dir:  {result['output_dir']}")
        print("=" * 60)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
