"""Class and normalization values from the training notebook."""
NUM_CLASSES = 6

CLASS_NAMES = ['Avicennia', 'Rhizophora', 'Mixed Mangrove',
               'Aquaculture', 'Buildings', 'Sea/River']

SEN_MEAN = [0.0352, 0.0444, 0.0366, 0.042, 0.0496, 0.0571, 0.0536, 0.0567, 0.0282, 0.0195, 0.0216, 0.0228, 0.0943, 0.0158, 0.098, 0.0024, 0.0371, 0.0464, 0.0403, 0.0456, 0.0512, 0.0581, 0.0547, 0.058, 0.0319, 0.023, 0.0228, 0.02, 0.078, 0.0061, 0.0751, 0.0036, 0.0356, 0.0431, 0.0319, 0.0357, 0.042, 0.0483, 0.0451, 0.0482, 0.0254, 0.018, 0.0029, 0.0132, 0.0661, 0.045]

SEN_STD  = [0.0458, 0.055, 0.0469, 0.0496, 0.0725, 0.0886, 0.0878, 0.0937, 0.0395, 0.0276, 0.2911, 0.1704, 0.1647, 0.3013, 0.2532, 0.2323, 0.0489, 0.0585, 0.0525, 0.0546, 0.0718, 0.0865, 0.0855, 0.0913, 0.0432, 0.0317, 0.2735, 0.1628, 0.1528, 0.2757, 0.227, 0.2166, 0.0502, 0.0583, 0.0442, 0.047, 0.0717, 0.0873, 0.0855, 0.0921, 0.0393, 0.0281, 0.2701, 0.1661, 0.1486, 0.2936]

PLANET_MEAN = [130.3377/255, 132.1437/255, 130.2918/255]

PLANET_STD  = [56.8424/255, 52.8236/255, 59.925/255 ]


SENTINEL_BAND_NAMES = ['B2_annual', 'B3_annual', 'B4_annual', 'B5_annual', 'B6_annual', 'B7_annual', 'B8_annual', 'B8A_annual', 'B11_annual', 'B12_annual', 'NDVI_annual', 'NDRE_annual', 'NDMI_annual', 'B2_dry', 'B3_dry', 'B4_dry', 'B5_dry', 'B6_dry', 'B7_dry', 'B8_dry', 'B8A_dry', 'B11_dry', 'B12_dry', 'NDVI_dry', 'NDRE_dry', 'NDMI_dry', 'B2_wet', 'B3_wet', 'B4_wet', 'B5_wet', 'B6_wet', 'B7_wet', 'B8_wet', 'B8A_wet', 'B11_wet', 'B12_wet', 'NDVI_wet', 'NDRE_wet', 'NDMI_wet', 'max_NDVI', 'min_NDVI', 'mean_NDVI', 'std_NDVI', 'NDVI_amplitude', 'NDVI_sum', 'valid_month_count']
