import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

batchSize = 1
semi = False
points = False

if semi:
    diag_path = ['results/diag_batch_random_bazinTD_batch' + str(batchSize) + '_vanilla.csv', 
                 'results/diag_batch_random_bazinTD_batch' + str(batchSize) + '.csv', 
                 'results/diag_batch_nlunc_bazinTD_batch' + str(batchSize) + '.csv',
                 'results/diag_batch_semi_bazinTD_batch' + str(batchSize) + '.csv']
else:
    diag_path = ['results/diag_batch_canonical_bazinTD_batch' + str(batchSize) + '.dat', 
                 'results/diag_batch_random_bazinTD_batch' + str(batchSize) + '.dat', 
                 'results/diag_batch_nlunc_bazinTD_batch' + str(batchSize) + '.dat']

acc = []
eff = []
pur = []
fom = []


for k in range(len(diag_path)):
    # read diagnostics
    op1 = open(diag_path[k], 'r')
    lin1 = op1.readlines()
    op1.close()

    data_str = [elem.split(',') for elem in lin1]
    acc.append(np.array([[float(data_str[k][-1]), float(data_str[k][2])] for k in range(batchSize, len(data_str) - batchSize, batchSize)]))
    eff.append(np.array([[float(data_str[k][-1]), float(data_str[k][3])] for k in range(batchSize, len(data_str) - batchSize, batchSize)]))
    pur.append(np.array([[float(data_str[k][-1]), float(data_str[k][4])] for k in range(batchSize, len(data_str) - batchSize, batchSize)]))
    fom.append(np.array([[float(data_str[k][-1]), float(data_str[k][-2])] for k in range(batchSize, len(data_str) - batchSize,batchSize)]))


acc = np.array(acc)
eff = np.array(eff)
pur = np.array(pur)
fom = np.array(fom)

t1 = acc[0][:,0] <= 80
t2 = acc[0][:,0] > 80

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

fig = plt.figure(figsize=(20, 14))
ax1 = plt.subplot(2,2,1)
ax1.fill_between(np.arange(15,80), 0, 1, color='grey', alpha=0.05)
if points:
    l0 = ax1.scatter(acc[0][:,0][t2], acc[0][:,1][t2], color='#dd0100', marker='v', s=70, label='Canonical')
    l1 = ax1.scatter(acc[1][:,0][t2], acc[1][:,1][t2], color='#fac901', marker='o', s=50, label='Passive Learning')
    ax1.scatter(acc[0][:,0][t1], acc[0][:,1][t1], color='#dd0100', marker='v', s=70, alpha=0.2)
    ax1.scatter(acc[1][:,0][t1], acc[1][:,1][t1], color='#fac901', marker='o', s=50, alpha=0.2)
    ax1.scatter(acc[2][:,0][t1], acc[2][:,1][t1], color='#225095', marker='^', s=70, alpha=0.2)
    if semi:
        l2 = ax1.scatter(acc[2][:,0][t2], acc[2][:,1][t2], color='#225095', marker='^', s=70, label='AL: N-least certain')
        l3 = ax1.scatter(acc[3][:,0][t2], acc[3][:,1][t2], color='#30303a', marker='s', s=50, label='AL: Semi-supervised')
        ax1.scatter(acc[3][:,0][t1], acc[3][:,1][t1], color='#30303a', marker='s', s=50, alpha=0.2)
    else:
        l2 = ax1.scatter(acc[2][:,0][t2], acc[2][:,1][t2], color='#225095', marker='^', s=70, label='AL: Uncertainty sampling')
else:
    l0 = ax1.plot(acc[0][:,0][t2], acc[0][:,1][t2], color='#dd0100', ls=':', lw=5.0, label='Canonical')
    l1 = ax1.plot(acc[1][:,0][t2], acc[1][:,1][t2], color='#fac901', ls='--', lw=5.0, label='Passive Learning')
    ax1.plot(acc[0][:,0][t1], acc[0][:,1][t1], color='#dd0100', ls=':', lw=5.0, alpha=0.2)
    ax1.plot(acc[1][:,0][t1], acc[1][:,1][t1], color='#fac901', ls='--', lw=5.0, alpha=0.2)
    ax1.plot(acc[2][:,0][t1], acc[2][:,1][t1], color='#225095', ls='-.', lw=5.0, alpha=0.2)
    if semi:
        l2 = ax1.plot(acc[2][:,0][t2], acc[2][:,1][t2], color='#225095', ls='-.', lw=5.0, label='AL: N-least certain') 
        l3 = ax1.plot(acc[3][:,0][t2], acc[3][:,1][t2], color='#30303a', lw=5.0, label='AL: Semi-supervised')
        ax1.plot(acc[3][:,0][t1], acc[3][:,1][t1], color='#30303a', lw=5.0, alpha=0.2)
    else:
        l2 = ax1.plot(acc[2][:,0][t2], acc[2][:,1][t2], color='#225095', ls='-.', lw=5.0, label='AL: Uncertainty sampling') 

#ax1.set_yticks(np.arange(0.4,1.0,0.1))
#ax1.set_yticklabels(np.arange(0.4,1.0,0.1), fontsize=22)
ax1.set_xticks(range(20,190,20))
ax1.set_xticklabels(range(20,190,20), fontsize=22)
ax1.set_xlabel('Survey duration (days)', fontsize=30)
ax1.set_ylabel('Accuracy', fontsize=30)
ax1.set_ylim(0.4,0.9)
ax1.set_xlim(15,183)

ax2 = plt.subplot(2,2,2)
ax2.fill_between(np.arange(15,80), 0, 1, color='grey', alpha=0.05)
if points:
    ax2.scatter(eff[0][:,0][t2], eff[0][:,1][t2], color='#dd0100', marker='v', s=70)
    ax2.scatter(eff[1][:,0][t2], eff[1][:,1][t2], color='#fac901', marker='o', s=50)
    ax2.scatter(eff[2][:,0][t2], eff[2][:,1][t2], color='#225095', marker='^', s=70)
    ax2.scatter(eff[0][:,0][t1], eff[0][:,1][t1], color='#dd0100', marker='v', s=70, alpha=0.2)
    ax2.scatter(eff[1][:,0][t1], eff[1][:,1][t1], color='#fac901', marker='o', s=50, alpha=0.2)
    ax2.scatter(eff[2][:,0][t1], eff[2][:,1][t1], color='#225095', marker='^', s=70, alpha=0.2)
    if semi:
        ax2.scatter(eff[3][:,0][t2], eff[3][:,1][t2], color='#30303a', marker='s', s=50)
        ax2.scatter(eff[3][:,0][t1], eff[3][:,1][t1], color='#30303a', marker='s', s=50, alpha=0.2)
else:
    ax2.plot(eff[0][:,0][t2], eff[0][:,1][t2], color='#dd0100', ls=':', lw=5.0)
    ax2.plot(eff[1][:,0][t2], eff[1][:,1][t2], color='#fac901', ls='--', lw=5.0)
    ax2.plot(eff[2][:,0][t2], eff[2][:,1][t2], color='#225095',  ls='-.', lw=5.0)
    ax2.plot(eff[0][:,0][t1], eff[0][:,1][t1], color='#dd0100', ls=':', lw=5.0, alpha=0.2)
    ax2.plot(eff[1][:,0][t1], eff[1][:,1][t1], color='#fac901', ls='--', lw=5.0, alpha=0.2)
    ax2.plot(eff[2][:,0][t1], eff[2][:,1][t1], color='#225095',  ls='-.', lw=5.0, alpha=0.2)
    if semi:
        ax2.plot(eff[3][:,0][t2], eff[3][:,1][t2], color='#30303a', lw=5.0)
        ax2.plot(eff[3][:,0][t1], eff[3][:,1][t1], color='#30303a', lw=5.0, alpha=0.2)

ax2.set_xlabel('Survey duration (days)', fontsize=30)
ax2.set_ylabel('Efficiency', fontsize=30)
ax2.set_xticks(range(20,190,20))
ax2.set_xticklabels(range(20,190,20), fontsize=22)
#ax2.set_yticks(np.arange(0,1.0,0.1))
#ax2.set_yticklabels(np.arange(0,1.0,0.1), fontsize=22)
ax2.set_xlim(15,183)
ax2.set_ylim(0, 0.7)

ax3 = plt.subplot(2,2,3)
ax3.fill_between(np.arange(15,80), 0, 1, color='grey', alpha=0.05)
if points:
    ax3.scatter(pur[0][:,0][t2], pur[0][:,1][t2], color='#dd0100', marker='v', s=70)
    ax3.scatter(pur[1][:,0][t2], pur[1][:,1][t2], color='#fac901', marker='o', s=50)
    ax3.scatter(pur[2][:,0][t2], pur[2][:,1][t2], color='#225095', marker='^', s=70)
    ax3.scatter(pur[0][:,0][t1], pur[0][:,1][t1], color='#dd0100', marker='v', s=70, alpha=0.2)
    ax3.scatter(pur[1][:,0][t1], pur[1][:,1][t1], color='#fac901', marker='o', s=50, alpha=0.2)
    ax3.scatter(pur[2][:,0][t1], pur[2][:,1][t1], color='#225095', marker='^', s=70, alpha=0.2)
    if semi:
        ax3.scatter(pur[3][:,0][t2], pur[3][:,1][t2], color='#30303a', marker='s', s=50)
        ax3.scatter(pur[3][:,0][t1], pur[3][:,1][t1], color='#30303a', marker='s', s=50, alpha=0.2)
else:
    ax3.plot(pur[0][:,0][t2], pur[0][:,1][t2], color='#dd0100', ls=':', lw=5.0)
    ax3.plot(pur[1][:,0][t2], pur[1][:,1][t2], color='#fac901', ls='--', lw=5.0)
    ax3.plot(pur[2][:,0][t2], pur[2][:,1][t2], color='#225095', ls='-.', lw=5.0)
    ax3.plot(pur[0][:,0][t1], pur[0][:,1][t1], color='#dd0100', ls=':', lw=5.0, alpha=0.2)
    ax3.plot(pur[1][:,0][t1], pur[1][:,1][t1], color='#fac901', ls='--', lw=5.0, alpha=0.2)
    ax3.plot(pur[2][:,0][t1], pur[2][:,1][t1], color='#225095', ls='-.', lw=5.0, alpha=0.2)
    if semi:
        ax3.plot(pur[3][:,0][t2], pur[3][:,1][t2], color='#30303a', lw=5.0)
        ax3.plot(pur[3][:,0][t1], pur[3][:,1][t1], color='#30303a', lw=5.0, alpha=0.2)
ax3.set_xlabel('Survey duration (days)', fontsize=30)
ax3.set_ylabel('Purity', fontsize=30)
ax3.set_xticks(range(20,190,20))
ax3.set_xticklabels(range(20,190,20), fontsize=22)
#ax3.set_yticks(np.arange(0, 1.1,0.2))
#ax3.set_yticklabels(np.arange(0, 1.1,0.2), fontsize=22)
ax3.set_xlim(15,183)
ax3.set_ylim(0, 1.0)

ax4 = plt.subplot(2,2,4)
ax4.fill_between(np.arange(15,80), 0, 1, color='grey', alpha=0.05)
if points:
    ax4.scatter(fom[0][:,0][t2], fom[0][:,1][t2], color='#dd0100', marker='v', s=70)
    ax4.scatter(fom[1][:,0][t2], fom[1][:,1][t2], color='#fac901', marker='o', s=50)
    ax4.scatter(fom[2][:,0][t2], fom[2][:,1][t2], color='#225095', marker='^', s=70)
    ax4.scatter(fom[0][:,0][t1], fom[0][:,1][t1], color='#dd0100', marker='v', s=70, alpha=0.2)
    ax4.scatter(fom[1][:,0][t1], fom[1][:,1][t1], color='#fac901', marker='o', s=50, alpha=0.2)
    ax4.scatter(fom[2][:,0][t1], fom[2][:,1][t1], color='#225095', marker='^', s=70, alpha=0.2)
    if semi:
        ax4.scatter(fom[3][:,0][t2], fom[3][:,1][t2], color='#30303a', marker='s', s=50)
        ax4.scatter(fom[3][:,0][t1], fom[3][:,1][t1], color='#30303a', marker='s', s=50, alpha=0.2)
else:
    ax4.plot(fom[0][:,0][t2], fom[0][:,1][t2], color='#dd0100', ls=':', lw=5.0)
    ax4.plot(fom[1][:,0][t2], fom[1][:,1][t2], color='#fac901', ls='--', lw=5.0)
    ax4.plot(fom[2][:,0][t2], fom[2][:,1][t2], color='#225095', ls='-.', lw=5.0)
    ax4.plot(fom[0][:,0][t1], fom[0][:,1][t1], color='#dd0100', ls=':', lw=5.0, alpha=0.2)
    ax4.plot(fom[1][:,0][t1], fom[1][:,1][t1], color='#fac901', ls='--', lw=5.0, alpha=0.2)
    ax4.plot(fom[2][:,0][t1], fom[2][:,1][t1], color='#225095', ls='-.', lw=5.0, alpha=0.2)
    if semi:
        ax4.plot(fom[3][:,0][t2], fom[3][:,1][t2], color='#30303a', lw=5.0)
        ax4.plot(fom[3][:,0][t1], fom[3][:,1][t1], color='#30303a', lw=5.0, alpha=0.2)
ax4.set_xlabel('Survey duration (days)', fontsize=30)
ax4.set_ylabel('Figure of merit', fontsize=30)
ax4.set_xticks(range(20,190,20))
ax4.set_xticklabels(range(20,190,20), fontsize=22)
#ax4.set_yticks(np.arange(0, 0.275, 0.05))
#ax4.set_yticklabels(np.arange(0, 0.275, 0.05), fontsize=22)
ax4.set_ylim(0, 0.25)
ax4.set_xlim(15,183)

handles, labels = ax1.get_legend_handles_labels()
ph = [plt.plot([], marker="", ls="")[0]]
h = ph + handles
l = ["Strategy:"] + labels
lgd = ax1.legend(h, l, loc='upper center', bbox_to_anchor=(1.025,1.295), ncol=5, fontsize=23.5)

plt.subplots_adjust(left=0.075, right=0.95, top=0.875, bottom=0.075, wspace=0.25, hspace=0.25)
plt.savefig('time_domain_batch' + str(batchSize) + '.png')

