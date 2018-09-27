""" Configuration parameters for pipeline
"""
import os.path

DATA_DIR = '/data'

# path to time domain directory
TIME_DOMAIN_DIR = os.path.join(DATA_DIR, 'time_domain')

# path to data for canonical strategy
CAN_PATHS = [os.path.join(DATA_DIR, 'vanilla_lightcurves.txt'),
             os.path.join(DATA_DIR, 'vanilla_labels.txt')]

# path to raw data
RAW_DIR = os.path.join(DATA_DIR, 'SIMGEN_PUBLIC_DES')

# tags for supernova classes
SN_IBC = ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16', '18',
          '22', '23', '29', '45', '28']
SN_II = ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25', '26',
         '27', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
         '40', '41', '42', '43', '44']

# DES filters
FILTERS = ['g', 'r', 'i', 'z']

# strategy options: canonical, random, nlunc
# random -> randomly draw elements from the test sample
# nlunc -> n-least uncertain or Active Learning
MODE = 'random'

# select initial training
TRAIN_IDX = []
TRAIN_LABEL = []

# results
OUT_DIR = os.path.join(DATA_DIR, 'results')
_DIAG_FNAME = 'diag_batch_' + MODE + '_bazinTD_batch1.dat'
DIAG_NAME = os.path.join(OUT_DIR, _DIAG_FNAME)

_QUERIES_FNAME = 'queries_batch_' + MODE + '_bazinTD_batch1.csv'
QUERIES_NAME = os.path.join(OUT_DIR, _QUERIES_FNAME)
