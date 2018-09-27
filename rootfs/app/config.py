# path to time domain directory
time_domain_dir = '/data/time_domain'

# path to data for canonical strategy
dir2 = '/data'
can_path = [dir2 + '/vanilla_lightcurves.txt', dir2 + '/vanilla_labels.txt']

# path to raw data
raw_dir = '/data/SIMGEN_PUBLIC_DES/'

# tags for supernova classes
snIbc = ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16', '18', '22',
         '23', '29', '45', '28']
snII = ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25', '26', '27',
        '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
        '42', '43', '44']

# DES filters
filters = ['g', 'r', 'i', 'z']

# strategy options: canonical, random, nlunc
# random -> randomly draw elements from the test sample
# nlunc -> n-least uncertain or Active Learning
mode = 'random'

# select initial training
train_indx = []
label_train = []

# results
out_dir = '/data/results'
diag_name = out_dir + '/diag_batch_' + mode + '_bazinTD_batch1.dat'
queries_name = out_dir + '/queries_batch_' + mode + '_bazinTD_batch1.csv'
