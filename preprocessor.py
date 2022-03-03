from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Preprocessor():
    def __init__(self, scaling_strategy, imput_strategy):
        assert scaling_strategy in ["minmax", "standard"]
        assert imput_strategy in ["median", "mode", "ind"]

        if scaling_strategy == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        if imput_strategy == "median":
            self.imput_nans = self._median_imput
        elif imput_strategy == "mode":
            self.imput_nans = self._mode_imput
        else:
            self.imput_nans = self._ind_imput

    @staticmethod
    def _median_imput(dataset):
        median = dataset.median()
        return dataset.fillna(median)

    @staticmethod
    def _mode_imput(dataset):
        mode = dataset.mode()[:1]
        return dataset.fillna(mode)

    @staticmethod
    def _ind_imput(dataset):
        inds = dataset.max() + 1
        return dataset.fillna(inds)

    def __call__(self, dataset):
        dataset = self.imput_nans(dataset)
        dataset = self.scaler.fit_transform(dataset)
        return dataset
