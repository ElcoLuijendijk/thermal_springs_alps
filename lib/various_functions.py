import numpy as np


def SI_geothermometers(S):
    
    """
    Equations for SiO2 geothermometers
    """
    
    maxT = np.zeros((9, S.shape[0]))
    
    # Fournier-1
    maxT[0] = (1309 / (5.19 - np.log10(S))) - 273.15
    
    # Fournier-2
    maxT[1] = (1522 / (5.75 - np.log10(S))) - 273.15
    
    # Fournier-Potter
    # min
    maxT[2] = -(42.198 - 1.345) + (0.28831 - 0.01337) * S \
              - (3.6686 * 1e-4 - 3.152 * 1E-5) * S**2 \
              + (3.1665 * 1e-7 -2.421* 1e-7) * S**3 \
              + (77.034 - 1.216) * np.log10(S)
    maxT[3] = -(42.198 + 1.345) + (0.28831 + 0.01337) * S \
              - (3.6686 * 1e-4 + 3.152 * 1E-5) * S**2 \
              + (3.1665 * 1e-7 + 2.421* 1e-7) * S**3 \
              + (77.034 + 1.216) * np.log10(S)
    
    # Verma-Santoyo-1
    maxT[4] = -(44.119 - 0.438) + (0.24469 - 0.00573) * S \
             - (1.7414 * 1e-4 - 1.365 * 1e-5) * S**2 \
             + (79.305 - 0.427) * np.log10(S)
                
    maxT[5] = -(44.119 + 0.438) + (0.24469 + 0.00573) * S \
             - (1.7414 * 1e-4 + 1.365 * 1e-5) * S**2 \
             + (79.305 + 0.427) * np.log10(S)
    
    # Vera-Santoyo-2
    #maxT[6] = 140.82 + (0.23517 - 0.00179) * S
    #maxT[7] = 140.82 + (0.23517 + 0.00179) * S
    
    # Arnorsson-2
    maxT[6] = -55.3 + 0.3659 * S - 5.3954 * 1e-4 * S**2 + 5.5132 * 1e-7 * S**3 + 74.360 * np.log10(S)
    
    # Verma
    maxT[7] = (1175.7 - 31.7) / (4.88 - 0.08 -np.log10(S)) - 273.15
    maxT[8] = (1175.7 + 31.7) / (4.88 + 0.08 -np.log10(S)) - 273.15
    #
    #maxT = (1175.7 / (4.88 - np.log10(S))) - 273.15
    
    min_circ_temp = np.min(maxT, axis=0)
    max_circ_temp = np.max(maxT, axis=0)
    mean_circ_temp = np.mean(maxT, axis=0)            
    
    return (mean_circ_temp, min_circ_temp, max_circ_temp, maxT)


