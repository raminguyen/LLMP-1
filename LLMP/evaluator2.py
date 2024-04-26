import LLMP as L
import re
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch

class Evaluator2:

    # Calculate mean squared error (MSE)
    def calculate_mse(gt, answers):
        gt_array = np.array(gt)
        answers_array = np.array(answers)

        return mean_squared_error(gt_array,answers_array)

    
    # Calculate midmean logistic absoulte error (MALE)
    def calculate_mlae(gt, answers):
        gt_array = np.array(gt)
        answers_array = np.array(answers)

        mlae = np.log2(mean_absolute_error(gt_array*100, answers_array*100) + .125)
        return mlae

    # Calculate mean
    def calculate_mean(answers):
        return np.mean(answers)

    # Calculate std
    def calculate_std(answers):
        return np.std(answers)

    @staticmethod
    def run(data, query, models):
        images = [d[0] for d in data]
        gt = [d[1] for d in data]
        results = {'gt': gt}

        for model_name in models:
            raw_answers = []
            parsed_answers = []
            forced_repetitions = 0
            times = []

            for image in images:
                torch.torch.cuda.empty_cache()
                FLAG = False
                start_time = time.time()

                while not FLAG:
                    answer = ""
                    match model_name:
                        case "LLaVA":
                            answer = L.LLaVA.query(query, image)
                        case "ChatGPT":
                            answer = L.ChatGPT.query(query, image)
                        case "CustomLLaVA":
                            answer = L.CustomLLaVA.query(query, image)
                        
                    #pattern = r'(?<![\d\w*.-])\d+(?:\.\d+)?(?:-(?:\d+(?:\.\d+)?))?(?![\d\w*.-])'
                    #matches = re.findall(pattern, answer)

                    values = re.findall(r'(\d+\.\d+)', answer)
                    

                    if (len(values) != 5):
                        values = values[-5:]

                    ranges_numbers = [float(val) for val in values]

                    if len(values) == 5:
                        raw_answers.append(answer)
                        parsed_answers.append(ranges_numbers)
                        FLAG = True
                        end_time = time.time()
                        times.append((end_time - start_time) * 1000)
                        if model_name == "ChatGPT":
                            time.sleep(2)  # Avoid hitting rate limits!
                    else:
                        forced_repetitions += 1
                        if model_name == "ChatGPT":
                            time.sleep(2)  # Avoid hitting rate limits!


                
            mse = Evaluator2.calculate_mse(gt, parsed_answers)
            mlae = Evaluator2.calculate_mlae(gt, parsed_answers)
            #mean = Evaluator2.calculate_mean(parsed_answers)
            #std = Evaluator2.calculate_std(parsed_answers)

            #mse = 0
            #mlae = 0
            mean = None
            std = None

            results[model_name] = {
                'parameters': None, 
                'raw_answers': raw_answers,
                'parsed_answers': parsed_answers,
                'mean': mean,
                'std': std,
                'mse': mse, 
                'mlae': mlae, 
                'times': times,
                'forced_repetitions': forced_repetitions
            }

        return results
