import sys
sys.path.insert(0, '.')

from src.orchestrator import DataScienceOrchestrator

o = DataScienceOrchestrator('data/titanic_sample.csv', 'demo_run')
result = o.run()

print('\n=== RESULT ===')
print(f"Status: {result['status']}")
print(f"Run: {result['run_name']}")
print(f"Experiment ID: {result['experiment_id']}")
print(f"Report: {result['report_path']}")
print(f"Output: {result['output_dir']}")
