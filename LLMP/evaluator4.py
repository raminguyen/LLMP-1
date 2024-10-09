import torch
import re
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

class Evaluator4:

    def __init__(self):
        self.results = None
    
    @staticmethod
    def calculate_mse(gt, answers):
        gt_array = np.array(gt).flatten()  # Flatten to ensure 1D array
        answers_array = np.array(answers).flatten()  # Flatten to ensure 1D array
        return mean_squared_error(gt_array, answers_array)

    @staticmethod
    def calculate_mlae(gt, answers):
        gt_array = np.array(gt).flatten()  # Flatten to ensure 1D array
        answers_array = np.array(answers).flatten()  # Flatten to ensure 1D array
        mlae = np.log2(mean_absolute_error(gt_array * 100, answers_array * 100) + 0.125)
        return mlae

    @staticmethod
    def calculate_mean(answers):
        return np.mean(answers)

    @staticmethod
    def calculate_std(answers):
        return np.std(answers)

    @staticmethod
    def parse_answer(answer):
        pattern = r'(?<![\d\w*.-])\d+(?:\.\d+)?(?:-(?:\d+(?:\.\d+)?))?(?![\d\w*.-])'
        matches = re.findall(pattern, answer)
        ranges_numbers = []

        for match in matches:
            if '-' in match:
                # Split into start and end of the range
                start, end = map(float, match.split('-'))
                ranges_numbers.append([start, end])
            else:
                # Single number
                ranges_numbers.append([float(match)])

        final_ranges = []
        i = 0
        while i < len(ranges_numbers):
            if len(ranges_numbers[i]) == 2:  # This is a range
                final_ranges.append(ranges_numbers[i])
            elif len(ranges_numbers[i]) == 1:
                # Handle pairing single numbers if they are followed by another single number
                if i + 1 < len(ranges_numbers) and len(ranges_numbers[i + 1]) == 1:
                    final_ranges.append([ranges_numbers[i][0], ranges_numbers[i + 1][0]])
                    i += 1  # Skip the next number since it's paired
                else:
                    final_ranges.append(ranges_numbers[i])
            i += 1
        
        return final_ranges

    def run(self, data, query, models):
        """Run experiments."""
        images = [d[0] for d in data]
        gt = [d[1] for d in data]
        results = {'gt': gt}

        for model_name, model_instance in models.items():
            results[model_name] = {}
            # Run three times to calculate STD
            mlae_list = []
            
            for i in range(3):
                raw_answers = []
                parsed_answers = []
                forced_repetitions = 0
                times = []
    
                for image in images:
                    torch.cuda.empty_cache()
                    FLAG = False
                    start_time = time.time()

                    while not FLAG:
                        answer = model_instance.query(query, image)
    
                        ranges_numbers = Evaluator4.parse_answer(answer)
    
                        if len(ranges_numbers) > 0:
                            raw_answers.append(answer)
    
                            midpoints = []
                            # If there are two ranges, calculate midpoints
                            if len(ranges_numbers) == 2:
                                midpoint_x = (ranges_numbers[0][0] + ranges_numbers[0][1]) / 2  # Average of the first elements
                                midpoint_y = (ranges_numbers[1][0] + ranges_numbers[1][1]) / 2  # Average of the second elements
                                midpoints.append(midpoint_x)
                                midpoints.append(midpoint_y)
                            elif len(ranges_numbers) == 1:  # If there is only one range
                                midpoints.append(ranges_numbers[0][0])  # Take the existing first value
                                midpoints.append(ranges_numbers[0][1])  # Take the existing second value
                                
                            parsed_answers.append(midpoints)
                            FLAG = True
                            end_time = time.time()
                            times.append((end_time - start_time) * 1000)
                        else:
                            forced_repetitions += 1

                if len(parsed_answers) == 0:
                    parsed_answers = [[0] for _ in images]
    
                mse = Evaluator.calculate_mse(gt_flat, midpoints_flat)
                mlae = Evaluator.calculate_mlae(gt_flat, midpoints_flat)
                mean = Evaluator.calculate_mean(midpoints_flat)

                mlae_list.append(mlae)

                results[model_name][f"run_{i}"] = {
                    'raw_answers': raw_answers,
                    'parsed_answers': parsed_answers,
                    'mean': mean,
                    'mse': mse,
                    'mlae': mlae,
                    'times': times,
                    'forced_repetitions': forced_repetitions
                }
                
            results[model_name]['average_mlae'] = Evaluator.calculate_mean(mlae_list)
            results[model_name]['std'] = Evaluator.calculate_std(mlae_list)
            results[model_name]['confidence'] = 1.96*bs.bootstrap(mlae_list, stat_func=bs_stats.std).value
        
        self.results = results

        return self.results

    def get_results(self):
            """Retrieve the stored results."""
            if self.results is None:
                raise ValueError("No results found. Run the 'run' method first.")
            return self.results

    def save_results(self, filename):
        """Save the results."""
        if self.results is None:
            raise ValueError("No results found. Run the 'run' method first.")

        with open(filename, 'w') as json_file:
            json.dump(self.results, json_file, indent=4) 
