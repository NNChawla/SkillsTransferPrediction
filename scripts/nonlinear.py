def calculate_lyapunov_exponent(velocity_series):
    data = velocity_series.drop_nulls().to_numpy()
    max_lag = 100
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    ac = np.array([np.sum((data[:n - lag] - mean) * (data[lag:] - mean)) / ((n - lag) * var)
                for lag in range(max_lag)])
    threshold_ac = 1 / np.e
    min_tstep = np.argmax(ac < threshold_ac)
    return nolds.lyap_r(data, min_tsep=min_tstep)

from .NONANLibrary.LyE_R import *
class NonlinearAnalysis:
    def __init__(self, velocity_series):
        self.velocity_series = velocity_series.drop_nulls().to_numpy()
        self.optimal_lag = None
        self.optimal_emb_dim = None
        self.min_tstep = None
        # self.init_parameters()

    def average_mutual_information(self, max_lag):
        ami = []
        # Convert velocity series to non-negative integers
        min_val = np.min(self.velocity_series)
        shifted_series = self.velocity_series - min_val  # Shift to make all values non-negative
        # Convert to integers (required by pyinform)
        discretized_series = np.floor(shifted_series * 1000).astype(int)  # Scale and convert to integers
        
        for lag in range(1, max_lag + 1):
            mi = mutual_info(discretized_series[:-lag], discretized_series[lag:], local=False)
            ami.append(mi)
        return np.array(ami)
    
    def autocorrelation(self, max_lag):
        n = len(self.velocity_series)
        mean = np.mean(self.velocity_series)
        var = np.var(self.velocity_series)
        ac = np.array([np.sum((self.velocity_series[:n - lag] - mean) * (self.velocity_series[lag:] - mean)) / ((n - lag) * var)
                    for lag in range(max_lag)])
        return ac
    
    def init_parameters(self):
        # max_lag = 100
        # ami_values = self.average_mutual_information(max_lag)
        # self.optimal_lag = np.argmin(ami_values) + 1

        # max_dim = 10
        # dims = np.arange(1, max_dim + 1)
        # fnn_percent = dimension.fnn(self.velocity_series, tau=self.optimal_lag, dim=dims)[0]
        # threshold = 10.0
        # self.optimal_emb_dim = dims[np.where(fnn_percent < threshold)[0][0]]

        max_ac_lag = 100
        ac_values = self.autocorrelation(max_ac_lag)
        threshold_ac = 1 / np.e
        self.min_tstep = np.argmax(ac_values < threshold_ac)

    def calculate_lyapunov_exponent(self):
        # return LyE_R(self.velocity_series, 100, self.optimal_lag, self.optimal_emb_dim)
        return nolds.lyap_r(self.velocity_series, min_tsep=self.min_tstep)
    