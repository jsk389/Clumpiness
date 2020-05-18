import numpy as np
import scipy

#from keras.utils import to_categorical

from xgboost import XGBModel

class ModelwithTemperature:
    """
    Want to wrap xgboost model with temperature scaling model
    """
    def __init__(self, model):
        super(ModelwithTemperature, self).__init__()
        self.model = model
        self.temperature = np.ones(1) * 1.5

    def forward(self, inputs):
        logits = self.model.predict(inputs, output_margin=True)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits, temperature=None):
        """
        Temperature scaling on logits
        """
        if temperature is None:
            return logits / self.temperature
        else:
            self.temperature = temperature
            return logits / self.temperature

    def _optimising_function(self, temperature, y_true, logits):
        """
        Function to minimise
        """
        return self._cross_entropy(y_true, self.temperature_scale(logits, temperature))

    def _cross_entropy(self, y_true, logits, epsilon=1e-12):
        """
        Calculate the cross-entropy loss - where y_pred here will be the logits

        >>> model = ModelwithTemperature()
        >>> predictions = np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.96])
        >>> targets = np.array([[0, 0, 0, 1], [0, 0, 0, 1])
        >>> model._cross_entropy(targets, np.log(predictions))
        0.71355817782
        """
        probs = scipy.special.softmax(logits, axis=1)
        preds = np.clip(probs, epsilon, 1-epsilon)
        return -np.sum(y_true * np.log(preds+epsilon))/logits.shape[0]        

    def set_temperature(self, inputs, labels):
        """
        Tune the temperature using the validation set by optimising the NLL
        """
        # Fetch logits
        logits = self.model.predict(inputs, output_margin=True)
        
        
        nll_criterion = self._cross_entropy
        ece_criterion = _ECELoss()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels)
        before_temperature_ece = ece_criterion(logits, labels)
        print('Before temperature scaling - NLL: {}, ECE: {}'.format(before_temperature_nll, before_temperature_ece))

        # 

        # Optimise the termpature w.r.t NLL
        optimizer = scipy.optimize.minimize(self._optimising_function, x0=self.temperature, args=(labels, logits))
        print(optimizer)
        # Calculate NLL and ECE after temperature scaling
        print("Optimal Temperature: {}".format(self.temperature))
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels)   
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels)
        print('After temperature scaling - NLL: {}, ECE: {}'.format(after_temperature_nll, after_temperature_ece))
        return self
    
    def predict(self, inputs):
        logits = self.model.predict(inputs, output_margin=True)
        return self.temperature_scale(logits)

class _ECELoss:
    """
    Calculate the Expected Calibration Error of a model.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        bin_boundaries = np.linspace(0, 1, n_bins+1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __call__(self, logits, labels):
        softmaxes = scipy.special.softmax(logits, axis=1)
        confidences, predictions = np.max(softmaxes, axis=1), np.argmax(softmaxes, axis=1)
        #predictions = to_categorical(predictions)
        labels = np.argmax(labels, axis=1)
        print(predictions, labels)
        accuracies = (predictions == labels) * 1

        ece = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower) & (confidences < bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
