import sys
sys.path.append('./LLMP/')
import ClevelandMcGill as C

class Image:
    @staticmethod
    def figure1(task):
        match task:
            case "angle":
                sparse, image, label, parameters = C.Figure1.angle()

            case "position_common_scale":
                sparse, image, label, parameters = C.Figure1.position_common_scale()

            case "length":
                sparse, image, label, parameters = C.Figure1.length([False, False, False])

            case "direction":
                sparse, image, label, parameters = C.Figure1.direction()

            case "area":
                sparse, image, label, parameters = C.Figure1.area()
        
            case "volume":
                sparse, image, label, parameters = C.Figure1.volume()

            case "curvature":
                sparse, image, label, parameters = C.Figure1.curvature()

            case "shading":
                sparse, image, label, parameters = C.Figure1.shading()

        return image
    
    @staticmethod
    def figure3(type):
        data, labels = C.Figure3.generate_datapoint()
        match type:
            case "bar":
                image = C.Figure3.data_to_barchart(data)
            case "pie":
                image = C.Figure3.data_to_piechart(data)

        return image
    

    @staticmethod
    def figure4(type):
        data, labels = C.Figure4.generate_datapoint()
        match type:
            case "type1":
                image = C.Figure4.data_to_type1(data)
            case "type2":
                image = C.Figure4.data_to_type2(data)
            case "type3":
                image = C.Figure4.data_to_type3(data)
            case "type4":
                image = C.Figure4.data_to_type4(data)
            case "type5":
                image = C.Figure4.data_to_type5(data)

        return image
    
    @staticmethod
    def figure12(framed):
        data, labels, parameters = C.Figure12.generate_datapoint()
        if framed:
            image = C.Figure12.data_to_framed_rectangles(data)
        else:
            image = C.Figure12.data_to_image(data)
        return image
    
    
    @staticmethod
    def weber(dots):
        match dots:
            case "10":
                image, label = C.Weber.base10()
            case "100":
                image, label = C.Weber.base100()
            case "1000":
                image, label = C.Weber.base1000()
        return image