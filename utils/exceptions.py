"""
Exception handling Module 
"""

class NeuralDispatcherError(Exception):
    pass

class DataLoadingError(NeuralDispatcherError):
    pass

class ModelLoadingError(NeuralDispatcherError):
    pass

class TrainingError(NeuralDispatcherError):
    pass

class InferenceError(NeuralDispatcherError):
    pass

class ModelSavingError(NeuralDispatcherError):
    pass

class DatasetError(NeuralDispatcherError):
    pass

class MetricsError(NeuralDispatcherError):
    pass