import LLMP as L
import re
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class Evaluator:

    # Calculate mean squared error (MSE)
    def calculate_mse(gt, answers):
        gt_array = np.array(gt)
        answers_array = np.array(answers)

        return mean_squared_error(gt_array,answers_array)

    
    # Calculate midmean logistic absoulte error (MALE)
    def calculate_mlae(gt, answers):
        gt_array = np.array(gt)
        answers_array = np.array(answers)

        male = np.log2(mean_absolute_error(gt_array*100, answers_array*100) + .125)
        return male

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
                FLAG = False
                start_time = time.time()

                while not FLAG:
                    answer = ""
                    match model_name:
                        case "LLaVA":
                            answer = L.LLaVA.query(query, image)
                        case "ChatGPT":
                            answer = L.ChatGPT.query(query, image)

                    range = re.findall(r'\d+', answer)
                    if range:
                        range_numbers = [int(r) for r in range]
                        raw_answers.append(answer)
                        parsed_answers.append(range_numbers)
                        FLAG = True
                        end_time = time.time()
                        times.append((end_time - start_time) * 1000)
                        time.sleep(8)  # Avoid hitting rate limits!
                    else:
                        forced_repetitions += 1
                        time.sleep(8)


            # Evaluation
            midpoints = [(a+b)/2 for a, b in parsed_answers]  # could be first or last
            mse = Evaluator.calculate_mse(gt, midpoints)
            mlae = Evaluator.calculate_mlae(gt, midpoints)

            results[model_name] = {
                'parameters': None, 
                'raw_answers': raw_answers,
                'parsed_answers': parsed_answers,
                'mse': mse, 
                'mlae': mlae, 
                'times': times,
                'forced_repetitions': forced_repetitions
            }

        print(results)
