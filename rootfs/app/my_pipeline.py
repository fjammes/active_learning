import config
from diagnostics import efficiency, purity, fom
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier


def _read_data(dir1, path_canonical=None):

    print('Reading data ... ')

    data = {}

    if path_canonical != None:
        op3 = open(path_canonical[0], 'r')
        lin3 = op3.readlines()
        op3.close()

        data3 = [elem.split() for elem in lin3[1:]]

        canonical_data = np.array(
            [[float(elem) for elem in item] for item in data3])

        op4 = open(path_canonical[1], 'r')
        lin4 = op4.readlines()
        op4.close()

        canonical_ids = [elem.split() for elem in lin4[1:]]
        canonical_labels = [int(line[2] != '0') for line in canonical_ids]

        indx_can = np.arange(len(canonical_ids))
        np.random.shuffle(indx_can)

        data['canonical_data'] = np.array(
            [canonical_data[i] for i in indx_can])
        data['canonical_labels'] = np.array(
            [canonical_labels[i] for i in indx_can])
        data['canonical_ids'] = np.array([canonical_ids[i] for i in indx_can])

    for day in range(20, 182):
        data[day] = {}

        op1 = open(dir1 + '/matrix_day_' + str(day) + '.dat', 'r')
        lin1 = op1.readlines()
        op1.close()

        data1 = [elem.split() for elem in lin1[1:]]

        matrix = np.array([[float(elem) for elem in line] for line in data1])

        op2 = open(dir1 + '/labels_day_' + str(day) + '.dat', 'r')
        lin2 = op2.readlines()
        op2.close()

        labels = [elem.split() for elem in lin2[1:]]

        train_data = []
        train_label = []
        train_ids = []
        test_data = []
        test_label = []
        test_ids = []

        for i in range(len(labels)):
            if labels[i][3] == 'False':
                test_data.append(matrix[i])
                test_label.append(int(labels[i][2] != '0'))
                test_ids.append(labels[i])
            else:
                train_data.append(matrix[i])
                train_label.append(int(labels[i][2] != '0'))
                train_ids.append(labels[i])

        data[day]['train_data'] = np.array(train_data)
        data[day]['train_labels'] = np.array(train_label)
        data[day]['test_data'] = np.array(test_data)
        data[day]['test_labels'] = np.array(test_label)
        data[day]['test_ids'] = test_ids
        data[day]['train_ids'] = train_ids

    print('DONE!\n')

    return data


# read data
data = _read_data(config.TIME_DOMAIN_DIR, path_canonical=config.CAN_PATHS)

# randomly draw 5 elements of the test (photometric) sample
# require that at least 1 of then is a Ia
train_indx = config.TRAIN_IDX
train_label = config.TRAIN_LABEL
while 1 not in train_indx and 0 not in train_label:
    train_indx = np.random.choice(
        range(data[20]['train_data'].shape[0]), size=5, replace=False)
    train_label = data[20]['train_labels'][train_indx]

matrix_train = data[20]['train_data'][train_indx]


diag = {}
queries = {}

for day in range(20, 181):

    count = 0

    # train RF
    clf = RandomForestClassifier(random_state=42, n_estimators=1000)
    clf.fit(matrix_train, train_label)

    # calculate probability of test data
    prob = clf.predict_proba(data[day]['test_data'])

    # predict types of test data
    prediction = clf.predict(data[day]['test_data'])

    # calculate diagnostics whe applied to test data
    acc = sum(prediction == data[day]['test_labels']) / \
        float(data[day]['test_data'].shape[0])
    eff = efficiency(prediction, data[day]['test_labels'])
    pur = purity(prediction, data[day]['test_labels'])
    fomr = fom(prediction, data[day]['test_labels'])

    print('---------------------------------')
    print('Results for day ' + str(day))
    print('\n')
    print('Size of training = ' + str(len(train_label)))
    print('\n')
    print('accuracy     =  ' + str(acc))
    print('efficiency   =  ' + str(eff))
    print('purity       =  ' + str(pur))
    print('fom          =  ' + str(fomr))
    print('\n')

    if day == 20:
        # store diagnostics
        diag[day] = [day - 20, '---', acc, eff, pur, fomr, day]
    else:
        diag[day] = [day - 20, queries[day - 1]
                     ['ID'][0], acc, eff, pur, fomr, day]

    if config.MODE == 'nlunc':
        # calculate probability of query data for next day
        prob_next = clf.predict_proba(data[day + 1]['train_data'])

        # predict types of query data for next day
        prediction_next = clf.predict(data[day + 1]['train_data'])

        # identify most uncertain object
        diff = [abs(line[0] - line[1]) for line in prob_next]
        indx_min = diff.index(min(diff))

    elif config.MODE == 'random':
        # passive learning gets random draws from those SN whose brightness allow a spectra to be taken
        indx_min = np.random.randint(
            low=0, high=data[day + 1]['train_data'].shape[0])

    elif config.MODE == 'canonical':
        # canonical just get the next bright object
        indx_min = count
        count = count + 1

    # store queried objects
    queries[day] = {}
    if config.MODE != 'canonical':
        queries[day]['data'] = data[day + 1]['train_data'][indx_min]
        queries[day]['label'] = data[day + 1]['train_labels'][indx_min]
        queries[day]['ID'] = data[day + 1]['train_ids'][indx_min]

    else:
        queries[day]['data'] = data['canonical_data'][indx_min]
        queries[day]['label'] = data['canonical_labels'][indx_min]
        queries[day]['ID'] = data['canonical_ids'][indx_min]

    # read query features
    f_op5 = os.path.join(config.RAW_DIR, queries[day]['ID'][0] + '.DAT')
    op5 = open(f_op5, 'r')
    lin5 = op5.readlines()
    op5.close()

    data5 = [elem.split() for elem in lin5]

    mag = {}
    mag['g'] = []
    mag['r'] = []
    mag['i'] = []
    mag['z'] = []

    SNR = {}
    SNR['g'] = []
    SNR['r'] = []
    SNR['i'] = []
    SNR['z'] = []

    for line in data5:
        if len(line) > 1 and line[0] == 'SIM_REDSHIFT:':
            queries[day]['z'] = line[1]
        if len(line) > 1 and line[0] == 'SNID:':
            queries[day]['snid'] = line[1]

        elif len(line) > 1 and line[0] == 'SIM_NON1a:':
            if line[1] in config.SN_IBC:
                queries[day]['sntype'] = 'Ibc'
            elif line[1] in config.SN_II:
                queries[day]['sntype'] = 'II'
            elif line[1] == '0':
                queries[day]['sntype'] = 'Ia'
            else:
                raise ValueError('SNTYPE not listed!')
            queries[day]['qclas'] = line[1]

        elif len(line) > 1 and line[0] == 'OBS:':
            mag[line[2]].append(float(line[7]))
            SNR[line[2]].append(float(line[6]))

    queries[day]['g_pkmag'] = min(mag['g'])
    queries[day]['r_pkmag'] = min(mag['r'])
    queries[day]['i_pkmag'] = min(mag['i'])
    queries[day]['z_pkmag'] = min(mag['z'])

    queries[day]['g_SNR'] = np.mean(SNR['g'])
    queries[day]['r_SNR'] = np.mean(SNR['r'])
    queries[day]['i_SNR'] = np.mean(SNR['i'])
    queries[day]['z_SNR'] = np.mean(SNR['z'])

    # add to training
    if config.MODE == 'canonical':
        matrix_train = list(matrix_train)
        matrix_train.append(data['canonical_data'][indx_min])
        matrix_train = np.array(matrix_train)

        train_label = list(train_label)
        train_label.append(data['canonical_labels'][indx_min])
        train_label = np.array(train_label)

    else:
        matrix_train = list(matrix_train)
        matrix_train.append(data[day + 1]['train_data'][indx_min])
        matrix_train = np.array(matrix_train)

        train_label = list(train_label)
        train_label.append(data[day + 1]['train_labels'][indx_min])
        train_label = np.array(train_label)


# train RF
clf = RandomForestClassifier(random_state=42, n_estimators=1000)
clf.fit(matrix_train, train_label)

# calculate probability of test data
prob = clf.predict_proba(data[day + 1]['test_data'])

# predict types of test data
prediction = clf.predict(data[day + 1]['test_data'])

# calculate diagnostics whe applied to test data
acc = sum(prediction == data[day + 1]['test_labels']) / \
    float(data[day + 1]['test_data'].shape[0])
eff = efficiency(prediction, data[day + 1]['test_labels'])
pur = purity(prediction, data[day + 1]['test_labels'])
fomr = fom(prediction, data[day + 1]['test_labels'])

diag[day + 1] = [day, queries[day]['ID'][0], acc, eff, pur, fomr, day + 1]

print('---------------------------------')
print('Results for day ' + str(day + 1))
print('\n')
print('Size of training = ' + str(len(train_label)))
print('\n')
print('accuracy     =  ' + str(acc))
print('efficiency   =  ' + str(eff))
print('purity       =  ' + str(pur))
print('fom          =  ' + str(fomr))
print('\n')


# save results
if not os.path.isdir(config.OUT_DIR):
    os.makedirs(config.OUT_DIR)

op3 = open(config.DIAG_NAME, 'w')
op3.write('nqueries,snid,acc,eff,pur,fom,day_of_survey\n')

for k in range(20, 182):
    for elem in diag[k][:-1]:
        op3.write(str(elem) + ',')
    op3.write(str(diag[k][-1]) + '\n')
op3.close()

op4 = open(config.QUERIES_NAME, 'w')
op4.write('nqueries,snid,sntype,z,g_pkmag,r_pkmag,i_pkmag,z_pkmag,g_SNR,r_SNR,i_SNR,z_SNR,qclas,day_of_survey\n')

for day in range(20, 181):
    op4.write(str(day - 19) + ',' + str(queries[day]['snid']) + ',' +
              queries[day]['sntype'] + ',' + queries[day]['z'] + ',')
    for f in config.FILTERS:
        op4.write(str(queries[day][f + '_pkmag']) + ',')
    for f in config.FILTERS:
        op4.write(str(queries[day][f + '_SNR']) + ',')
    op4.write(queries[day]['qclas'] + ',' + str(day) + '\n')
op4.close()
