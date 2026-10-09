<<<PATCH 01>>>
<<<OLD>>>
if mean_nll > 700: # ln(float_max) approx 709
<<<NEW>>>
if mean_nll > np.log(np.finfo(np.float64).max):
<<<END>>>