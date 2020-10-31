"""
#################################
# Configuration File
#################################
"""

Config_General = {'NUM_UAV': 2, 'NUM_EPS': 200, 'NUM_STEP': 30000, 'NUM_PKT': 1,
                  'Location_SaveFile': 1, 'CSI_SaveFile': 1, 'PlotLocation': 1, 'DF': 0, 'printFlag': 1,
                  'PlotResult': 0, 'NUM_RUN': 10}

Config_Param = {'T': 1, 'Noise': 1, 'Etha1': 1e-13, 'Etha2':1, 'Etha3': 1, 'Gamma_punish1': 0.1, 'Gamma_punish2': 0.4,
                'lambda1': 2, 'lambda2': 2, 'lambda3': 0.4, 'lambda3_3': 0.1, 'Sigmoid_coef': 7}

Size = 10
Config_Dim = {'Height': 10, 'Length': Size, 'Width': Size, 'UAV_L_MAX': Size, 'UAV_L_MIN': 0, 'UAV_W_MIN': 0,
              'UAV_W_MAX': Size, 'Divider': 10}

Config_Power = {'Power_fusion': 10, 'Power_source': 20, 'Power_pt': 10, 'Power_UAV_pr': 20, 'Energy': 10}

pathH = 'HMatrix%d' % Config_General.get('NUM_UAV')
pathDist = 'LocMatrix%d' % Config_General.get('NUM_UAV')
Config_Path = {'PathH': pathH, 'PathDist': pathDist}
