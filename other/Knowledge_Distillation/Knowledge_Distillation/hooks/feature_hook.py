class FeatureHook:
    def __init__(self):
        self.features = []

    def __call__(self, module, input, output):
        self.features.append(output)

    def clear(self):
        self.features.clear()
