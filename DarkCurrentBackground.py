# -*- coding: utf-8 -*-
"""
Dark Current Background
Paula Pérez Cobo
"""
import pandas as pd
import ast
from scipy.stats import poisson

#CCD parameters
npix = 16 * 6144
num_imgs = 3984
num_ccds = 4
eff_mask = 0.9219 #masking efficiency

#Lambdas 1e-, 2e-, >3e-
lambda_epiximg_1 = 2.81e-4
lambda_epiximg_2 = 5.18e-4
lambda_epiximg_3 = 3.92e-3

def poisson_product(pattern):
    lambdas = []
    for charge in pattern:
        if charge == 1:
            lambdas.append(lambda_epiximg_1)
        elif charge == 2:
            lambdas.append(lambda_epiximg_2)
        else:
            lambdas.append(lambda_epiximg_3)
    return prod([poisson.pmf(k, lam) for k, lam in zip(pattern, lambdas)])

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

def calculate_bkg_dc(detected_pattern, efficiencies_df):
    """
    Parameters
    ----------
    detected_pattern : string
        Pattern studied
    efficiencies_df : DataFrame
        Dataframe containing the pattern efficiencies

    Returns
    -------
    bkg_dc : float
        Total dark current background count for pattern detected_pattern.
    princp_term : float
        Principal term of the background: detected = origin pattern
    missclass : float
        Misidentification term of the background: detected =! origin pattern

    """
    df = efficiencies_df[efficiencies_df["detected_pattern"] == str(detected_pattern)]

    princp_term = 0
    missclass = 0

    for _, row in df.iterrows():
        origin = ast.literal_eval(row["origin_pattern"])
        detected = ast.literal_eval(row["detected_pattern"])
        eff = row["Efficiency_Mean"]

        poisson_val = poisson_product(origin)
        contrib = poisson_val * eff * npix * num_imgs * num_ccds * eff_mask

        if origin == detected:
            princp_term += contrib
        else:
            missclass += contrib

    bkg_dc = princp_term + missclass
    return bkg_dc, princp_term, missclass

# Read efficiencies
csv_path = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Diffusion Probabilities\Patterns\Alpha 10_7 iteraciones\EfficienciesBkgDC_AddedPatterns_mean.csv"
efficiencies_df = pd.read_csv(csv_path)
efficiencies_df["origin_pattern"] = efficiencies_df["origin_pattern"].apply(lambda x: x.strip('"'))
efficiencies_df["detected_pattern"] = efficiencies_df["detected_pattern"].apply(lambda x: x.strip('"'))

# Patterns from 2 to 5 electrons
patterns_to_evaluate = [
    (1,1), (2,1), (1,1,1), (3,1), (2,1,1), (2,2),
    (3,2), (4,1), (2,2,1), (3,1,1)
]

#Patterns from 2 to 4 electrons
# patterns_to_evaluate = [
#     (1,1), (2,1), (1,1,1), (3,1), (2,1,1), (2,2)
# ]

results = []

for pattern in patterns_to_evaluate:
    bkg_dc, princp_term, missclass = calculate_bkg_dc(pattern, efficiencies_df)
    results.append({
        "Pattern": pattern,
        "Bkg_DC": bkg_dc,
        "Principal": princp_term,
        "Missclassified": missclass
    })

results_df = pd.DataFrame(results)
print(results_df)

output_path = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Diffusion Probabilities\Patterns\BkgDC_4ccds.csv"
results_df.to_csv(output_path, index=False)

######################## LAMBDA OBTENTION #####################################

# n_events_2e = 188                   
# n_ccds = 4                          
# n_images = 3984                     
# n_pix = 6144 * 16    
# eff_2 = 0.96927
# eff_mask  = 0.9219     

# target = n_events_2e / (n_ccds * n_images * n_pix* eff_2*eff_mask)

# def poisson_2e_lambda(lambda_2):
#     return (lambda_2**2 / 2) * np.exp(-lambda_2) - target

# result = root_scalar(poisson_2e_lambda, bracket=[1e-8, 1e-3], method='brentq')

# if result.converged:
#     print(f"Lambda 2e- (e⁻/pix/img) ≈ {result.root:.2e}")
# else:
#     print("No se pudo encontrar la solución")


# n_events_3e = 14                                    
# eff_3 = 0.96935                  

# target = n_events_3e / (n_ccds * n_images * n_pix * eff_3 * eff_mask)

# def poisson_3e_lambda(lambda_):
#     return (lambda_**3 / 6) * np.exp(-lambda_) - target

# result = root_scalar(poisson_3e_lambda, bracket=[1e-8, 1e-2], method='brentq')

# if result.converged:
#     print(f"Lambda 3e- (e⁻/pix/img) ≈ {result.root:.2e}")
# else:
#     print("No se pudo encontrar la solución")