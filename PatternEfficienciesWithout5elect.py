"""
Pattern efficiencies for patterns with total charge Qe = [2,4]e-
Paula Pérez Cobo
"""

import numpy as np
from scipy.stats import norm
import csv
import os


sigma_res = 0.16

# PATTERN IDENTIFICATION VARIABLE
def compute_p(cdf_values, threshold=4.0):
    """
    Compute minimum pattern identification variable using precomputed 
    CDF values and compare with the threshold given.

    Parameters
    ----------
    cdf_values : list
        list containing all possible permutations of the cdf values computed
        for each pixel charge for the corresponding pattern
    threshold : float
        threshold for the pattern identification variable

    Returns
    -------
    bool
        Whether the pattern fulfills the required condition or not

    """
    return min(-np.log(np.prod(cdf_values_set)) for cdf_values_set in cdf_values) < threshold


def classify_pattern(complete_pattern_readen, sigma_res=0.16):
    """
    Pattern identification algorithm based on the computation of the 
    pattern identification variable. 

    Parameters
    ----------
    complete_pattern_readen : list
        pattern detected
    sigma_res : float
        readout noise

    Returns
    -------
    detected_patterns : list
        list containing the patterns detected

    """
    detected_patterns = []
    qmin = 3.75*sigma_res # minimum charge to consider a pixel charged
    while len(complete_pattern_readen) >= 3:
        sum_pattern_readen = sum(complete_pattern_readen[1:3]) 
        # if the total pattern charge > 5e- it is not selected
        if sum_pattern_readen > 5 + qmin:
            complete_pattern_readen = complete_pattern_readen[3:]
            continue     
        #Pattern and single event must be isolated: initial pixel in row
        if complete_pattern_readen[0] >= qmin:
            complete_pattern_readen = complete_pattern_readen[1:]
            continue
        if complete_pattern_readen[1] >= qmin:
            #First compute all possible cdfs:
            cdf_11 = norm.cdf(complete_pattern_readen[1], loc=1, scale=sigma_res)
            cdf_12 = norm.cdf(complete_pattern_readen[1], loc=2, scale=sigma_res)
            cdf_13 = norm.cdf(complete_pattern_readen[1], loc=3, scale=sigma_res)
            cdf_14 = norm.cdf(complete_pattern_readen[1], loc=4, scale=sigma_res)
            cdf_15 = norm.cdf(complete_pattern_readen[1], loc=5, scale=sigma_res)
    
            if len(complete_pattern_readen) >= 3:   
                cdf_21 = norm.cdf(complete_pattern_readen[2], loc=1, scale=sigma_res)
                cdf_22 = norm.cdf(complete_pattern_readen[2], loc=2, scale=sigma_res)
                cdf_23 = norm.cdf(complete_pattern_readen[2], loc=3, scale=sigma_res)
                cdf_24 = norm.cdf(complete_pattern_readen[2], loc=4, scale=sigma_res)
                cdf_25 = norm.cdf(complete_pattern_readen[2], loc=5, scale=sigma_res)
    
            if len(complete_pattern_readen) >= 5:
                cdf_31 = norm.cdf(complete_pattern_readen[3], loc=1, scale=sigma_res)
                cdf_32 = norm.cdf(complete_pattern_readen[3], loc=2, scale=sigma_res)
                cdf_33 = norm.cdf(complete_pattern_readen[3], loc=3, scale=sigma_res)
                cdf_34 = norm.cdf(complete_pattern_readen[3], loc=4, scale=sigma_res)
            
            #SINGLE EVENTS
            if compute_p([[cdf_11]], threshold=3.5):
                if  compute_p([[cdf_12]], threshold=3.5):
                    if  compute_p([[cdf_13]], threshold=3.5):
                        if  compute_p([[cdf_14]], threshold=3.5):
                            #isolated single event: final pixel
                            if complete_pattern_readen[2] < qmin: 
                                detected_patterns.append((4,)) # Single event (4,)
                                complete_pattern_readen = complete_pattern_readen[2:]
                                continue
                        else:
                            #isolated single event: final pixel
                            if complete_pattern_readen[2] < qmin:                                
                                detected_patterns.append((3,)) # Single event (3,)
                                complete_pattern_readen = complete_pattern_readen[2:]
                                continue
                    else:
                        #isolated single event: final pixel
                        if complete_pattern_readen[2] < qmin:  
                            detected_patterns.append((2,)) # Single event (2,)
                            complete_pattern_readen = complete_pattern_readen[2:]
                            continue
                else:
                    #isolated single event: final pixel
                    if complete_pattern_readen[2] < qmin:                              
                        detected_patterns.append((1,)) # Single event (1,)
                        complete_pattern_readen = complete_pattern_readen[2:]
                        continue
                    
            #PATTERNS        
            if len(complete_pattern_readen) >= 4:
                #Patterns must be composed of q > 3.75*sigma_res
                if complete_pattern_readen[2] >= qmin:          
                    sum_pattern_readen = sum(complete_pattern_readen[1:4])
                    if sum_pattern_readen > 5 + qmin:
                        complete_pattern_readen = complete_pattern_readen[4:]
                        continue
                    
                    if compute_p([[cdf_11, cdf_21]]): #11
                        if compute_p([[cdf_12, cdf_21], [cdf_11, cdf_22]]): #21
                            if compute_p([[cdf_13, cdf_21], [cdf_11, cdf_23]]): #31
                                if complete_pattern_readen[3] < qmin: #31 isolated
                                    detected_patterns.append((3, 1))
                                    complete_pattern_readen = complete_pattern_readen[3:]
                                    continue
                            else:
                                if len(complete_pattern_readen) >= 5 and complete_pattern_readen[3] > qmin:
                                    if compute_p([[cdf_12, cdf_21, cdf_31],[cdf_11, cdf_22, cdf_31],[cdf_11, cdf_21, cdf_32]], threshold=5.5): #211
                                        if complete_pattern_readen[4] < qmin: #isolation
                                            detected_patterns.append((2, 1, 1)) #211 isolated
                                            complete_pattern_readen = complete_pattern_readen[4:]
                                            continue
                                else:
                                    if compute_p([[cdf_12, cdf_21], [cdf_11, cdf_22]]):
                                        if compute_p([[cdf_12, cdf_22]]):
                                            if complete_pattern_readen[3] < qmin:  #22 isolated 
                                                detected_patterns.append((2, 2))
                                                complete_pattern_readen = complete_pattern_readen[3:]
                                                continue
                                        if complete_pattern_readen[3] < qmin:    #21 isolated
                                            detected_patterns.append((2, 1))
                                            complete_pattern_readen = complete_pattern_readen[3:]
                                            continue
                        else: 
                            if len(complete_pattern_readen) >= 5 and compute_p([[cdf_11, cdf_21, cdf_31]], threshold=5.5): #111
                                if compute_p([[cdf_12, cdf_21, cdf_31],[cdf_11, cdf_22, cdf_31],[cdf_11, cdf_21, cdf_32]],threshold=5.5): #211
                                    if complete_pattern_readen[4] < qmin: 
                                        detected_patterns.append((2, 1, 1)) #211 isolated
                                        complete_pattern_readen = complete_pattern_readen[5:]
                                        continue
                                else:  
                                    if complete_pattern_readen[4] < qmin:  
                                        detected_patterns.append((1, 1, 1)) #111 isolated
                                        complete_pattern_readen = complete_pattern_readen[5:]
                                        continue
                            else:
                                if complete_pattern_readen[3] < qmin: 
                                    detected_patterns.append((1, 1)) #11 isolated
                                    complete_pattern_readen = complete_pattern_readen[3:]
                                    continue        
            
        complete_pattern_readen = complete_pattern_readen[1:]
    
    return detected_patterns

################### PATTERN  EFFICIENCIES FOR BKG DARK CURRENT  ###############

save_directory = r'C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Diffusion Probabilities\Patterns'
file_name = "EfficienciesBkgDC_NotAdded.csv"
file_path = os.path.join(save_directory, file_name)

# List of possible origin patterns
complete_patterns = {
    (0,1,1,0): (1,1),
    (0,2,1,0): (2,1),
    (0,1,1,1,0): (1,1,1),
    (0,3,1,0): (3,1),
    (0,2,1,1,0): (2,1,1),
    (0,2,2,0): (2,2),
    (0,1,0,0): (1,),
    (0,2,0,0): (2,),
    (0,3,0,0): (3,),
    (0,4,0,0): (4,)
}

detected_patterns_to_track = [
    (1,1), (2,1), (1,1,1), (3,1), (2,1,1),
    (2,2)
]

sigma_res = 0.16

def compute_efficiencies(Niter=nit):
    """
    Computes the efficiencies for the complete set of possible patterns
    generated to be detected as detected_pattern

    Parameters
    ----------
    Niter : int
        Number of iterations

    Returns
    -------
    rows : list
        list containing the efficiency of detecting a pattern detected_pattern
        generated by another pattern, called origin_pattern

    """
    rows = []

    for complete_pattern, pattern_label in complete_patterns.items():
        count_detected = {}
        
        for _ in range(Niter):
            readen_pattern = np.random.normal(loc=complete_pattern, scale=sigma_res)
            detected = classify_pattern(readen_pattern, sigma_res=sigma_res)

            for d in detected:
                sorted_d = tuple(sorted(d, reverse=True))
                if sorted_d in detected_patterns_to_track:
                    count_detected[sorted_d] = count_detected.get(sorted_d, 0) + 1

        for detected_pattern, count in count_detected.items():
            efficiency = count / Niter
            rows.append((str(pattern_label), str(detected_pattern), efficiency))

    return rows

efficiency_rows = compute_efficiencies(Niter=10000000)

os.makedirs(save_directory, exist_ok=True)

with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["origin_pattern", "detected_pattern", "efficiency"])
    writer.writerows(efficiency_rows)

print(f"File saved in: {file_path}")

####################  PATTERN EFFICIENCIES FOR RATES ##########################

import numpy as np
from scipy.stats import norm
import csv
import os

# UNITS
um = 1.0
cm = 10000 * um
eV = 1.0
sigma_res = 0.16

# PARAMETERS FOR THE DIFFUSION MODEL
DIFF_A = 803.25 * um * um
DIFF_b = 6.5e-4 / um
DIFF_alpha = 1.0
DIFF_beta = 0 / eV

# CCD GEOMETRY
PIX_THICKNESS = 15.0 * um
ROWS = (0 * um, 1600 * PIX_THICKNESS * um)
COLS = (0 * um, 6144 * PIX_THICKNESS * um)
DEPTH = (0 * um, 670 * um)
BINNING_FACTOR = 100

save_directory = r'C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Diffusion Probabilities\Patterns'
os.makedirs(save_directory, exist_ok=True)

def probabilities_patterns(Ee, Niter=5000, binning=False):
    """
    A number of e-h is generated in a random z position and then diffused into 
    the CCD array with a lateral spread sigma_xy. Then, the pixels are binned
    and the patterns are readen in rows. Readout noise and the pattern 
    identification algorithm (classify_pattern()) are applied.
    
    The efficiencies are defined as the number of detected patterns for each 
    number of e-h detected in Niter realizations
    
    Parameters
    ----------
    Ee : float
        transferred energy
    Niter : int
        number of iterations
    binning : int
        pixel binning

    Returns
    -------
    dict
        dictionary containing the different patterns efficiencies for each
        number of e-h simulated

    """
    neh = np.arange(1, 11)  # neh range
    interaction_probabilities = {}
    # Patterns studied
    allowed_patterns = [(1,), (2,), (3,), (1, 1), (2, 1), (3, 1), (2, 2), (1, 1, 1), (2, 1, 1)]  
    charge_data = {(1,): [], (2,): [], (3,): [], (1,1): [], (2,1): [], (3,1): [], (2,2): [], (1,1,1): [], (2,1,1): []}
    def calculate_sigma_xy(z_positions, Ee):
        return np.sqrt(-DIFF_A * np.log(1 - DIFF_b * z_positions)) * (DIFF_alpha + DIFF_beta * Ee)
    
    for n in neh:
        pattern_counts = {}  
        positions = np.random.uniform(
            low=[ROWS[0], COLS[0], DEPTH[0]],
            high=[ROWS[1], COLS[1], DEPTH[1]],
            size=(Niter, 3))
        sigma_xy = calculate_sigma_xy(positions[:, 2], Ee)

        row_patterns = {}

        for i in range(Niter):
            if i % (Niter // 20) == 0:
                print(f"Progress: {100 * i / Niter:.1f}%")
            pos_diff = np.random.normal(
                loc=positions[i, :2],
                scale=sigma_xy[i],
                size=(n, 2))

            pixel_indices = [(int(pos[0] // PIX_THICKNESS), 
                              int(pos[1] // PIX_THICKNESS)) for pos in pos_diff]
            
            unique_pixels = {}
            for px in pixel_indices:
                if binning:
                    binned_row = px[0] // BINNING_FACTOR  
                    binned_px = (binned_row, px[1])  # Binning
                else:
                    binned_px = px  # Without binning
                
                if binned_px in unique_pixels:
                    unique_pixels[binned_px] += 1
                else:
                    unique_pixels[binned_px] = 1 

            row_distribution = {}
            for (row, col), count in unique_pixels.items():
                if row not in row_distribution:
                    row_distribution[row] = {}
                row_distribution[row][col] = count
            
            for row, cols in row_distribution.items():
                sorted_cols = sorted(cols.keys())
                min_col = sorted_cols[0]

                complete_pattern = [0]
                prev_col = min_col
                complete_pattern.append(cols[prev_col])

                for col in sorted_cols[1:]:
                    if col == prev_col + 1:
                        complete_pattern.append(cols[col])
                    else:
                        complete_pattern.extend([0] * (col - prev_col - 1))
                        complete_pattern.append(cols[col])
                    prev_col = col
                
                complete_pattern.append(0)
                
            # READOUT NOISE
            complete_pattern_readen = np.random.normal(loc=complete_pattern, scale=sigma_res)
            # PATTERN IDENTIFICATION ALGORITHM
            detected_patterns = classify_pattern(complete_pattern_readen, sigma_res=0.16)

            for detected_pattern in detected_patterns:
                sorted_pattern = tuple(sorted(detected_pattern, reverse=True))
                row_patterns[sorted_pattern] = row_patterns.get(sorted_pattern, 0) + 1
        print(f"Progress: 100.0% - neh {n} complete.")           
        for pattern, count in row_patterns.items():
            if pattern in allowed_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + count
        
        for pattern, count in pattern_counts.items():
            prob_int = count / Niter
            if pattern not in interaction_probabilities:
                interaction_probabilities[pattern] = {}
            interaction_probabilities[pattern][n] = prob_int

    return interaction_probabilities, charge_data


int_prob, charge_data = probabilities_patterns(0, Niter=10000000, binning=True)

output_file = os.path.join(save_directory, 'EfficienciesPatterns_NotAdded.csv')
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Pattern', 'neh', 'Efficiency'])
    
    for pattern, neh_values in int_prob.items():
        for neh, efficiency in neh_values.items():
            writer.writerow([pattern, neh, efficiency])

print(f"Efficiencies saved in {output_file}")