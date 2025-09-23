# Raise an exception to return to the top caller with mitra_n1_predictions
class MitraPredictionReturn(Exception):
    def __init__(self, value):
        self.value = value