# Benchmark Tool: Compare accuracy/time across methods

from pipeline.pipeline import SteganalysisPipeline

class BenchmarkTool:
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths
        self.results = []

    def run_all_methods_on_dataset(self):
        """Run all pipelines on each image in dataset."""
        # TODO: Time each method and record accuracy and relevant metrics
        pass

    def generate_comparison_report(self):
        """Generate table showing performance metrics."""
        # TODO: Print or export comparison report
        pass