from .ClevelandMcGill import Figure1
from .ClevelandMcGill import Figure3
from .ClevelandMcGill import Figure4
from .ClevelandMcGill import Figure12
from .ClevelandMcGill import Weber


class GPImage:
    @staticmethod
    def figure1(task):
        match task:
            case "angle":
                sparse, image, label, parameters = Figure1.angle()

            case "position_common_scale":
                sparse, image, label, parameters = Figure1.position_common_scale()

            case "length":
                sparse, image, label, parameters = Figure1.length([False, False, False])

            case "direction":
                sparse, image, label, parameters = Figure1.direction()

            case "area":
                sparse, image, label, parameters = Figure1.area()
        
            case "volume":
                sparse, image, label, parameters = Figure1.volume()

            case "curvature":
                sparse, image, label, parameters = Figure1.curvature()

            case "shading":
                sparse, image, label, parameters = Figure1.shading()

        return image, label
    
    @staticmethod
    def figure3(type):
        data, labels = Figure3.generate_datapoint()
        match type:
            case "bar":
                image = Figure3.data_to_barchart(data)
            case "pie":
                image = Figure3.data_to_piechart_aa(data)

        return image, labels
    

    @staticmethod
    def figure4(type):
        data, labels = Figure4.generate_datapoint()
        match type:
            case "type1":
                image = Figure4.data_to_type1(data)
            case "type2":
                image = Figure4.data_to_type2(data)
            case "type3":
                image = Figure4.data_to_type3(data)
            case "type4":
                image = Figure4.data_to_type4(data)
            case "type5":
                image = Figure4.data_to_type5(data)

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
        match dots:
            case "10":
                image, label = Weber.base10()
            case "100":
                image, label = Weber.base100()
            case "1000":
                image, label = Weber.base1000()
        return image, label
