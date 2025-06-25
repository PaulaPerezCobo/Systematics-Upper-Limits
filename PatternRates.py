# -*- coding: utf-8 -*-
"""
Code for generating each pattern rate
Paula Pérez Cobo
"""
##############################################################################
#########################   QEdark: dR/dE|E_e   ##############################
##############################################################################
"""
Number of events per kg year for a given Ee

Taken from QEdark: https://doi.org/10.1007/JHEP05(2016)046
"""
import sys
import numpy as np
import os
import csv
import ast  
import pandas as pd

dataDir = "C:\\Users\\Asus\\Documents\\MÁSTER\\TFM\\QEdark"
sys.path.append(dataDir) 

from QEdark_constants import *
from DM_halo_dist import *

rho_X = 0.3e9 # eV/cm^3
dQ = .02*alpha*me_eV #eV
dE = 0.1 # eV
wk = 2/137

## import QEdark data
nq = 900
nE = 500

fcrys = {'Si': np.transpose(np.resize(np.loadtxt(dataDir+'/Si_f2.txt',skiprows=1),(nE,nq))),
         'Ge': np.transpose(np.resize(np.loadtxt(dataDir+'/Ge_f2.txt',skiprows=1),(nE,nq)))}

"""
    materials = {name: [Mcell #eV, Eprefactor, Egap #eV, epsilon #eV, fcrys]}
    N.B. If you generate your own fcrys from QEdark, please remove the factor of "wk/4" below. 
"""

materials = {'Si': [2*28.0855*amu2kg, 2.0, 1.2, 3.8,wk/4*fcrys['Si']], \
             'Ge': [2*72.64*amu2kg, 1.8, 0.7, 2.8,wk/4*fcrys['Ge']]}

def FDM(q_eV,n):
    """
    DM form factor
    n = 0: FDM=1, heavy mediator
    n = 1: FDM~1/q, electric dipole
    n = 2: FDM~1/q^2, light mediator
    """
    return (alpha*me_eV/q_eV)**n

def mu_Xe(mX):
    """
    DM-electron reduced mass
    """
    return mX*me_eV/(mX+me_eV)

def dRdE(material, mX, Ee, FDMn, halo, params):
    """
    differential scattering rate for sigma_e = 1 cm^2
    at a fixed electron energy Ee
    given a DM mass, FDM, halo model
    returns dR/dE [events/kg/year]
    """
    n = FDMn
    if Ee < materials[material][2]: 
        print(f"Energy Ee is less than Egap: {materials[material][2]}") # check if less than Egap
        return 0
    else:
        qunit = dQ
        Eunit = dE
        Mcell = materials[material][0]
        Eprefactor = materials[material][1]
        Ei = int(np.floor(Ee*10)) # eV
        prefactor = ccms**2*sec2year*rho_X/mX*1/Mcell*alpha*me_eV**2 / mu_Xe(mX)**2
        array_ = np.zeros(nq)
        for qi in range(1,nq+1):
            q = qi*qunit
            vmin = (q/(2*mX)+Ee/q)*ccms
            if vmin > (vesc+vE)*1.1: # rough estimate for kinematicaly allowed regions
                array_[qi-1] = 0
            else:
                """
                define halo model
                """
                if halo == 'shm':
                    eta = etaSHM(vmin,params) # (cm/s)^-1 
                elif halo == 'tsa':
                    eta = etaTsa(vmin,params)
                elif halo == 'dpl':
                    eta = etaDPL(vmin,params)
                elif halo == 'msw':
                    eta = etaMSW(vmin,params)
                elif halo == 'debris':
                    eta = etaDF(vmin,params)
                else:
                    print("Undefined halo parameter. Options are ['shm','tsa','dpl','msw','debris']")
                """
                define array
                """
                array_[qi-1] = Eprefactor*(1/q)*eta*FDM(q,n)**2*materials[material][4][qi-1, Ei-1]
        return prefactor*np.sum(array_, axis=0) # [events (kg-year)^-1]


##############################################################################
#####################   Ionization: P(neh|E_e)   ############################
##############################################################################
"""
Probabilities of having a certain number of e-h given a certain Ee
Probabilities taken from: https://doi.org/10.1103/PhysRevD.102.063026
"""

# IMPORT PROBABILITIES
data = np.load("C:/Users/Asus/Documents/MÁSTER/TFM/p100K.npz")['data'].T

PAIR_CREATION_PROBABILITIES = {
        'E': data[0,:], #E_g <= E <= 50eV
        'P': data[1:,:],
        'Npair_bins': np.arange(1,21)
        }

##############################################################################
#####################   Pattern probs: P(pattern|neh, Ee)   #####################
##############################################################################
"""
Probabilities of having a certain pattern given a certain neh due to diffusion
"""

def load_efficiencies_patterns(filepath):
    efficiencies = {}
    with open(filepath, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            pattern = ast.literal_eval(row[0])
            neh = int(row[1])
            efficiency = float(row[2])
            if pattern not in efficiencies:
                efficiencies[pattern] = {}          
            efficiencies[pattern][neh] = efficiency
    
    return efficiencies

pattern_prob_file = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Diffusion Probabilities\Patterns\Alpha 10_7 iteraciones\EfficienciesPatternsAddedPatterns_alpha1_mean.csv"
pattern_probabilities = load_efficiencies_patterns(pattern_prob_file)


##############################################################################
#####################   Load rates dR/dE   ###################################
##############################################################################
rates_file = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Rate files\RatesUltralight_vE_253_7.csv"
rates_df = pd.read_csv(rates_file)
rates_df.columns = ['mX', 'Ee', 'Rate']
rates_df.set_index(['mX', 'Ee'], inplace=True)

# Round Ee
precision = 4
rates_df = rates_df.reset_index()
rates_df['Ee'] = rates_df['Ee'].round(precision)
rates_df.set_index(['mX', 'Ee'], inplace=True)

##############################################################################
#####################   Generate pattern rate file   #########################
##############################################################################

m_pix_g = 3.5072325e-5
t_pix_d = 0.02
nbinx = 1
nbiny = 100
mediator = "massless"

# Diffusion parameters
alpha_diff = 1.0
beta = 0
A = 803.25
b = 6.5e-4

# Velocity parameters
v0 = 238e5 # cm/s
vE = 253.7e5 #cm/s 
vesc = 544e5 #cm/s
vparams = [v0, vE, vesc]

save_path = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Rate files\Correct rates"

#DM mass range
mass_range = [7.54374000e+05,8.00000000e+05,8.48575800e+05,9.00000000e+05,
1.00000000e+06,1.20000000e+06,1.63780000e+06,2.00000000e+06,2.68269000e+06,
3.50000000e+06,4.39397000e+06,5.50000000e+06,7.19680000e+06,9.00000000e+06,
1.17876800e+07,1.50000000e+07,1.93069770e+07,3.16227700e+07,5.17947000e+07,
8.48342898e+07,1.38949500e+08,2.27584593e+08,3.72759372e+08,6.10540230e+08,
1.00000000e+09]


#Patterns with total charge from 2 to 5 e-: 11 21 111 31 211 22 32 41 221 311
studied_patterns = [(1, 1), (2, 1), (1, 1, 1), (3, 1), (2, 1, 1), (2, 2), (3, 2), (4, 1), (2, 2, 1), (3, 1, 1)]

E_bins = PAIR_CREATION_PROBABILITIES['E']
output_data = []

# === For rates not previously computed === 
 
# for mX in mass_range:   
#     dR_dE_vals = [dRdE('Si', mX, Ee, 2, 'shm', vparams) / (1000 * 365.25) for Ee in E_bins]
#     for pattern in studied_patterns:  
#         rate_pattern = 0
#         rate_background_pattern = 0
#         for i, Ee in enumerate(E_bins):
#             rate = dR_dE_vals[i]
#             rate_background = 1.5e-6 # 1.5e-5 * dE (0.1eV)
#             if rate == 0:
#                 continue
#             prob_ion = PAIR_CREATION_PROBABILITIES['P'][:, np.searchsorted(PAIR_CREATION_PROBABILITIES['E'], Ee)]
#             sum_prob_pattern = sum(prob_ion[neh - 1] * pattern_probabilities.get(pattern, {}).get(neh, 0) for neh in range(1, 11))
#             rate_pattern += rate * sum_prob_pattern #CAMBIADO A PATTERN
#             rate_background_pattern += rate_background * sum_prob_pattern
#         output_data.append([mX, mediator, nbinx, nbiny, m_pix_g, t_pix_d, A, b, alpha_diff, beta, vparams, pattern, rate_pattern, rate_background_pattern, "events / g day"])
#         print(f"pattern: {pattern} written")
#     print(f"mx: {mX} written")

# === Using previously computed rates ===

for mX in mass_range:
    dR_dE_vals = {
            Ee: rates_df.loc[(mX, Ee), 'Rate'] if (mX, Ee) in rates_df.index else 0.0
            for Ee in E_bins
            }
    for pattern in studied_patterns:
        rate_pattern = 0
        for Ee in E_bins:
            rate = dR_dE_vals[Ee]
            if rate == 0:
                continue
            prob_ion = PAIR_CREATION_PROBABILITIES['P'][:, np.searchsorted(PAIR_CREATION_PROBABILITIES['E'], Ee)]
            # Sum probability of the pattern across neh
            sum_prob_pattern = sum(
                prob_ion[neh - 1] * pattern_probabilities.get(pattern, {}).get(neh, 0)
                for neh in range(1, 11)
            )

            rate_pattern += rate * sum_prob_pattern

        output_data.append([
            mX, mediator, nbinx, nbiny, m_pix_g, t_pix_d, A, b,
            alpha_diff, beta, vparams, pattern, rate_pattern, "events / g day"
        ])
        print(f"pattern: {pattern} written")
    print(f"mx: {mX} written")
    
# Save in a CSV file
output_filename = os.path.join(save_path, "rates_neh_10_alpha_1_added_patterns.csv")
header = ['mX_eV', 'mediator', 'nbinx', 'nbiny', 'm_pix_g', 't_pix_d', 'A', 'b', 'alpha', 'beta', 'vparams', 'pattern', 'rate_pattern', 'units']


with open(output_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(output_data)

print(f"Data successfully written to {output_filename}")

