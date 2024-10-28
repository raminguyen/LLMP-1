import torch
import re
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import pandas as pd
import os
from PIL import Image

class Evaluator:
    
    def __init__(self):
        self.results = None
    
    @staticmethod
    def calculate_mse(gt, answers):
        """Calculate mse."""
        gt_array = np.array(gt).flatten()  # Flatten to ensure 1D array
        answers_array = np.array(answers).flatten()  # Flatten to ensure 1D array
        return mean_squared_error(gt_array, answers_array)

    @staticmethod
    def calculate_mlae(gt, answers):
        """Calculate mlae."""
        gt_array = np.array(gt).flatten()  # Flatten to ensure 1D array
        answers_array = np.array(answers).flatten()  # Flatten to ensure 1D array
        mlae = np.log2(mean_absolute_error(gt_array, answers_array) + 0.125)
        return mlae

    @staticmethod
    def calculate_mean(answers):
        """Calculate mean."""
        return np.mean(answers)

    @staticmethod
    def calculate_std(answers):
        """Calculate std."""
        return np.std(answers)

    @staticmethod
    def parse_answer(answer):
        """Parse a given string to find numbers only at the end of the sentence."""
        # Matches digits that appear before the end of a sentence or end of the string
        pattern = r'(\d+(?:\.\d+)?)(?=\s*[.!?]?$)'
        matches = re.findall(pattern, answer)

        # Convert matched numbers to floats
        ranges_numbers = [float(match) for match in matches]

        if len(ranges_numbers) == 0:
            return []
        
        return ranges_numbers

    def run(self, data, query, models):
        """Run experiments."""
        results = {'gt': [d[1] for d in data], 'image_path': [d[0] for d in data]}  # Capture ground truth and image paths

        for model_name, model_instance in models.items():
            results[model_name] = {}
            mlae_list = []
            
            for i in range(3):  # Repeat each experiment 3 times
                raw_answers = []
                parsed_answers = []
                forced_repetitions = 0
                times = []

                for image_path, ground_truth in data:
                    torch.cuda.empty_cache()
                    FLAG = False
                    start_time = time.time()

                    # Load the image from the path
                    image = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale

                    while not FLAG:
                        answer = model_instance.query(query, image)
                        ranges_numbers = Evaluator.parse_answer(answer)

                        if len(ranges_numbers) > 0:
                            raw_answers.append(answer)
                            parsed_answers.append(ranges_numbers)
                            FLAG = True
                            end_time = time.time()
                            times.append((end_time - start_time) * 1000)
                        else:
                            forced_repetitions += 1

                if len(parsed_answers) == 0:
                    parsed_answers = [[0] for _ in data]

                midpoints = [(sum(sublist) / 2) if len(sublist) > 1 else sublist[0] for sublist in parsed_answers]

                gt_flat = [item for sublist in results['gt'] for item in (sublist if isinstance(sublist, list) else [sublist])]
                midpoints_flat = (midpoints * (len(gt_flat) // len(midpoints) + 1))[:len(gt_flat)]

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
            results[model_name]['confidence'] = 1.96 * bs.bootstrap(np.array(mlae_list), stat_func=bs_stats.std).value
            
        self.results = results

        return self.results

    def save_results_csv(self, filename="results.csv"):
        """Transform all results for all tasks into a single DataFrame and save as a CSV in EXP1-Results folder."""
        if self.results is None:
            raise ValueError("No results found. Run the 'run' method first.")
        
        # Prepare data to store
        data = []

        for model_name, model_data in self.results.items():
            if model_name in ['gt', 'image_path']:  # Skip ground truth and image path keys for now
                continue

            for run_key, run_data in model_data.items():
                if run_key.startswith("run_"):
                    for idx, time in enumerate(run_data['times']):
                        # Gather data from each run, including image path, raw answers, parsed answers, and ground truth
                        data.append({
                            'model_name': model_name,
                            'run': run_key,
                            'image_path': self.results['image_path'][idx],  # Image path for each run
                            'ground_truth': self.results['gt'][idx],       # Ground truth for each run
                            'raw_answers': run_data['raw_answers'][idx],
                            'parsed_answers': run_data['parsed_answers'][idx],
                            'mean': run_data['mean'],
                            'mse': run_data['mse'],
                            'mlae': run_data['mlae'],
                            'forced_repetitions': run_data['forced_repetitions'],
                            'time_ms': time
                        })

            # Add aggregated metrics (average mlae, std, confidence)
            data.append({
                'model_name': model_name,
                'run': 'average',
                'image_path': None,
                'ground_truth': None,
                'raw_answers': None,
                'parsed_answers': None,
                'mean': None,
                'mse': None,
                'mlae': model_data.get('average_mlae'),
                'forced_repetitions': None,
                'time_ms': None,
                'std': model_data.get('std'),
                'confidence': model_data.get('confidence')
            })

        # Convert all collected data into a single DataFrame
        df = pd.DataFrame(data)
        
        # Define the directory path for saving results
        results_folder = os.path.join(os.getcwd(), "EXP1-Results")
        os.makedirs(results_folder, exist_ok=True)  # Ensure the results folder exists

        # Define the file path for saving results in the specified folder
        file_path = os.path.join(results_folder, filename)

        # Save the DataFrame to a single CSV file in the specified folder
        df.to_csv(file_path, index=False)
        
        # Return the DataFrame as well
        return df
