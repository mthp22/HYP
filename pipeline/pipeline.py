# Central implementation  for all methods
from analysers.lsboptimised import detect_stego as lsb_detect
from analysers.dct import analyze_image as dct_analyze
from analysers.cnn import predict_image as cnn_predict

class SteganalysisPipeline:
    def __init__(self):
        self.methods = {
            'LSB': lsb_detect,
            'DCT': dct_analyze,
            'CNN': cnn_predict
        }

    def run_analysis(self, method_name, file_path):
        """Run selected detection method on given image."""
        if method_name not in self.methods:
            raise ValueError("Unsupported method")
        return self.methods[method_name](file_path)

    def compare_results(self, results_list):
        """Compare results from multiple methods."""
        # TODO: Implement comparison logic
        pass