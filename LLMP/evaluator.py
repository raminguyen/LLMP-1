import torch
import re
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Evaluator:

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
                ranges_numbers.extend(match.split('-'))
                ranges_numbers = ranges_numbers[-2:]
            else:
                ranges_numbers.append(match)
        
        if len(ranges_numbers) == 0:
            return []
        
        return [float(r) for r in ranges_numbers]

    @staticmethod
    def run(data, query, models):
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
                    parsed_answers = [[0] for _ in images]
    
                midpoints = [(sum(sublist) / 2) if len(sublist) > 1 else sublist[0] for sublist in parsed_answers]
    
                # Ensure that both gt and predictions are consistent in length
                if isinstance(gt, (float, int, np.float64)):  # Check for single float or integer
                    gt = [[gt]]  # Convert single float or integer to list of lists
                elif isinstance(gt, list) and isinstance(gt[0], (float, int, np.float64)):
                    gt = [[value] for value in gt]  # Convert flat list of floats or integers to list of lists
    
                gt_flat = [item for sublist in gt for item in sublist]  # Flatten ground truth
                midpoints_flat = (midpoints * (len(gt_flat) // len(midpoints) + 1))[:len(gt_flat)]  # Replicate and slice
    
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
            
        
        return results
