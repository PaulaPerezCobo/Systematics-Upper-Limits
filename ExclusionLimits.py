# -*- coding: utf-8 -*-
"""
Exclusion limits code
Paula Pérez Cobo
"""

import math
import numpy as np
import pandas as pd
import os
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# DARK CURRENT BACKGROUND

file_path = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Diffusion Probabilities\Patterns\BkgDC_4ccds.csv"
df = pd.read_csv(file_path)

df['Pattern'] = df['Pattern'].astype(str)

# Define the desired pattern order
pattern_order = ["(1, 1)", "(2, 1)", "(1, 1, 1)", "(3, 1)", "(2, 1, 1)", "(2, 2)", 
                    "(3, 2)", "(4, 1)", "(2, 2, 1)", "(3, 1, 1)"]

# pattern_order = ["(1, 1)", "(2, 1)", "(1, 1, 1)", "(3, 1)", "(2, 1, 1)", "(2, 2)"]

Bkg_dark = [df.loc[df['Pattern'] == pattern, 'Bkg_DC'].values[0] for pattern in pattern_order]
print(f"Dark Current Background: {Bkg_dark}")

# DATA
#11 21 111 31 211 22 32 41 221 311

Data0 = [114,0,0,0,0,0,0,0,0,0]
# Data0 = [2740,0,1,0,0,0,0,0,0,0]


# RATES
fname = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Rate files\Correct rates\rates_neh_10_alpha_1_added_patterns.csv"
df = pd.read_csv(fname)

mX_eV_arr = df["mX_eV"].unique()

# PREPARE SIGNAL AND BACKGROUND
def prepareSignal(ip):
    mX_eV = mX_eV_arr[ip]
    signal_data = df[df["mX_eV"] == mX_eV].rate_pattern.to_numpy()
    
    Signal = signal_data *  979.35 * 10**-35
    Bkg = Bkg_dark 
    
    return Signal, Bkg

################################################################################
################################################################################
"""
Exclusion limits methodology
Taken from DAMIC-M Collaboration
"""

def muHat(x):
    val = sum(data[i] * Sref[i] / (x * Sref[i] + Bkg[i]) - Sref[i] for i in range(A, B))
    return val 

def T_mu(n, mu, s, b):
    nu = mu * s + b
    nu_hat = Mu_hat * s + b
    if Mu_hat > mu:
        return 0
    if Mu_hat <= 0:
        nu_hat = b
    if nu_hat == 0:
        return 2 * nu
    return -2 * (n * math.log(nu) - nu - (n * math.log(nu_hat) - nu_hat))

def Prob(muS):
    tmodel = np.zeros(N)
    count = 0
    tData = 0  
    precision = math.sqrt(N*CL*(1-CL))*3
    for i in range(A,B):
        if Sref[i] > 0:
            tData += T_mu(data[i], muS, Sref[i], Bkg[i])
    for i in range(A,B):
        if Sref[i] > 0:
            poisson_samples = np.random.poisson(muS*Sref[i] + Bkg[i], N)
            tmodel += [T_mu(n, muS, Sref[i], Bkg[i]) for n in poisson_samples]
    count = sum(1 for x in tmodel if x >= tData)
    val = round((count-N*(1-CL))/precision)*precision
    return val

N = 10000 # Number of simulations
Npattern = 10 #Number of patterns
CL = 0.9 #Confidence level
A = 0
B = Npattern
data = Data0
xMass_list = []
xLim_list = []

for ip in range(0, 25):
    Sref, Bkg = prepareSignal(ip)
    if muHat(0) < 0:
        Mu_hat = 0
        muLim = 1000 * data[0] / Sref[0] 
    else:
        Mu_hat = brentq(muHat, 0, 10 * data[0] / Sref[0]) 
        muLim = 1000 * Mu_hat    

    root = brentq(Prob, 0, muLim)
    xMass_list.append(mX_eV_arr[ip] / 1E6)
    xLim_list.append(root * 10**(-35))
    print(f"Limit for {mX_eV_arr[ip] / 1E6:.2e} MeV calculated")

print("-------------------------------------------------------------")    

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(xMass_list, xLim_list, color='black', lw=2, label='DAMIC-M, This work')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$m_\chi$(MeV/c$^2$)')
plt.ylabel(r'$\overline_{\sigma}_e (cm$^2$)')
plt.legend()
plt.show()

##############################################################################
##############################################################################

# Save limits in a file
save_path = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Exclusion Limits\exclusion_limits.txt"
new_column_header = "Limit 1"

if os.path.exists(save_path):
    file_exc = pd.read_csv(save_path, delim_whitespace=True)
else:
    file_exc = pd.DataFrame(columns=["DM Mass (MeV)"])

new_data = pd.DataFrame({"DM Mass (MeV)": xMass_list, new_column_header: xLim_list})

if "DM Mass (MeV)" in file_exc.columns:
    file_exc = pd.merge(file_exc, new_data, on="DM Mass (MeV)", how="outer")
else:
    file_exc = new_data

file_exc.to_csv(save_path, sep=" ", index=False)

print(f"File saved in: {save_path}")


################### LIMITS APPLYING GAUSSIAN FLUCTUATIONS #####################


# output_limit_file = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Exclusion Limits\exclusion_limits_lambda_times5_104ccds.txt"
# output_data0_file = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Exclusion Limits\data_fluctuations_lambda_times5_104ccds.csv"


# if os.path.exists(output_limit_file):
#     file_exc = pd.read_csv(output_limit_file, delim_whitespace=True)
# else:
#     file_exc = pd.DataFrame(columns=["DM Mass (MeV)"])

# data0_record = []
# DataR = np.array([67398.48,166.88,90.80,59.16,0.24,0.38,0.14,0.28,0,0])
# # 2% Fluctuation
# alpha = 0.02

# for run in range(45, 50):
#     sigma = alpha * DataR
#     sigma[DataR == 0] = 0
#     Data0 = np.random.normal(DataR, sigma)
#     Data0 = np.clip(Data0, 0, None)
#     data = Data0

#     record = {'Run': f"limit {run}"}
#     for i, pat in enumerate(pattern_order):
#         record[pat] = Data0[i]
#     data0_record.append(record)

#     xMass_list = []
#     xLim_list = []

#     try:
#         for ip in range(0, 25):
#             Sref, Bkg = prepareSignal(ip)
#             if muHat(0) < 0:
#                 Mu_hat = 0
#                 muLim = 1000 * data[0] / Sref[0]
#             else:
#                 Mu_hat = brentq(muHat, 0, 10 * data[0] / Sref[0])
#                 muLim = 1000 * Mu_hat

#             root = brentq(Prob, 0, muLim)
#             xMass_list.append(mX_eV_arr[ip] / 1E6)
#             xLim_list.append(root * 10**(-35))
#             print(f"Limit {run} for {mX_eV_arr[ip] / 1E6:.2e} MeV calculated")

#         # Save limit results
#         new_column_header = f"limit {run}"
#         new_data = pd.DataFrame({"DM Mass (MeV)": xMass_list, new_column_header: xLim_list})
#         if "DM Mass (MeV)" in file_exc.columns:
#             file_exc = pd.merge(file_exc, new_data, on="DM Mass (MeV)", how="outer")
#         else:
#             file_exc = new_data

#         # Plot Results
#         plt.figure(figsize=(10, 6))
#         plt.plot(xMass_list, xLim_list, color='black', lw=2, label='DAMIC-M, This work')
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel(r'$m_\chi$(MeV/c$^2$)')
#         plt.ylabel(r'$\overline_{\sigma}_e (cm$^2$)')
#         plt.legend()
#         plt.show()
#         file_exc.to_csv(output_limit_file, sep=" ", index=False)
        
#         print(f"Run {run} saved to: {output_limit_file}")

#         df_data0 = pd.DataFrame(data0_record)
#         df_data0.to_csv(output_data0_file, index=False)
#         print(f"Fluctuations up to run {run} saved to: {output_data0_file}")

#     except Exception as e:
#         print(f"Error in run {run}: {e}. Skipping to next run.")
#         continue

