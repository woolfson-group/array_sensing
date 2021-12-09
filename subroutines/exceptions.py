


class PlateLayoutError(Exception):
    pass


class FluorescenceSaturationError(Exception):
    pass


class NaNFluorescenceError(Exception):
    pass


class MinMaxFluorescenceError(Exception):
    pass


class AlgorithmError(Exception):
    pass

def create_generator(n):
    yield(['' for n in range(n)], []) 
