from .ClevelandMcGill import Figure1
from .ClevelandMcGill import Figure3
from .ClevelandMcGill import Figure4
from .ClevelandMcGill import Figure12
from .ClevelandMcGill import Weber
import numpy as np

class GPImage:
    @staticmethod
    def figure1(task):
        if task == "angle":
            sparse, image, label, parameters = Figure1.angle()
        elif task == "position_common_scale":
            sparse, image, label, parameters = Figure1.position_common_scale()
        elif task == "position_non_aligned_scale":
            sparse, image, label, parameters = Figure1.position_non_aligned_scale()
        elif task == "length":
            sparse, image, label, parameters = Figure1.length([False, False, False])
        elif task == "direction":
            sparse, image, label, parameters = Figure1.direction()
        elif task == "area":
            sparse, image, label, parameters = Figure1.area()
        elif task == "volume":
            sparse, image, label, parameters = Figure1.volume()
        elif task == "curvature":
            sparse, image, label, parameters = Figure1.curvature()
        elif task == "shading":
            sparse, image, label, parameters = Figure1.shading()
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return image, label
    
    @staticmethod
    def figure3(type):
        data, labels = Figure3.generate_datapoint()
        if type == "bar":
            image = Figure3.data_to_barchart(data)
        elif type == "pie":
            image = Figure3.data_to_piechart(data)
        else:
            raise ValueError(f"Unknown type: {type}")
        
        return image, labels
    
    @staticmethod
    def figure4(type):
        data, labels = Figure4.generate_datapoint()
        current_max = 93 - data[0] - data[1]
        
        while current_max / 3.0 <= 4:
            data, labels = Figure4.generate_datapoint()
            current_max = 93 - data[0] - data[1]
        
        if type == "type1":
            image = Figure4.data_to_type1(data)
        elif type == "type2":
            image = Figure4.data_to_type2(data)
        elif type == "type3":
            image = Figure4.data_to_type3(data)
        elif type == "type4":
            image = Figure4.data_to_type4(data)
        elif type == "type5":
            image = Figure4.data_to_type5(data)
        else:
            raise ValueError(f"Unknown type: {type}")
        
        return image, labels
    
    @staticmethod
    def figure12(framed):
        data, labels, parameters = Figure12.generate_datapoint()
        if framed:
            image = Figure12.data_to_framed_rectangles(data)
        else:
            image = Figure12.data_to_bars(data)
        return image, labels
    
    @staticmethod
    def weber(dots):
        if dots == "10":
            image, label = Weber.base10()
        elif dots == "100":
            image, label = Weber.base100()
        elif dots == "1000":
            image, label = Weber.base1000()
        else:
            raise ValueError(f"Unknown dots: {dots}")
        
        return image, label
