from glob import glob
import matplotlib.pyplot as plt
import numpy as np

f1 = plt.figure()
colors = ['r', 'b', 'm', 'y', 'k', 'g', 'r']

# Another set of sublots for the per class counts and accuracies

for num_file, filename in enumerate(glob('log/*.log')):
    performances = []
    perclass_performances = []
    counts_total = []
    with open(filename) as f:
        policy_name = 'unknown'
        for line in f:
            # loop over all the lines and use the data for the respective plots
            if 'policyname' in line:
                policy_name = line.split('---')[-1].strip()
            if 'INFO' in line:
                if 'PERFORMANCE' in line:
                    elements = line.split('---')
                    step = int(elements[-2])
                    performance = float(elements[-1])

                    performances.append((step, performance))
                if 'PERCLASS' in line:
                    elements = line.split('---')
                    step = int(elements[-2])
                    perclass_string = elements[-1]
                    perclass_row = [float(p) for p in perclass_string.split('--')]
                    perclass_performances.append([step] + perclass_row)
                if 'COUNTS' in line:
                    elements = line.split('---')
                    step = int(elements[-2])
                    counts_string = elements[-1]
                    counts_row = [int(p) for p in counts_string.split('--')]
                    counts_total.append([step] + counts_row)

    performances = np.array(performances)
    perclass_performances = np.array(perclass_performances)
    counts_total = np.array(counts_total)

    # Lots of pyplot magic
    plt.plot(performances[:, 0], performances[:, 1], label=policy_name, c=colors[num_file])
    plt.xlabel('time')
    plt.ylabel('performance')
plt.legend()


f_c, axarr_c = plt.subplots(1, 10)

largest_count = np.max(counts_total)
print(largest_count)
for num, ax in enumerate(axarr_c):
    ax.plot(counts_total[:, 0], counts_total[:, num + 1], c='r', label='Total counts')
    ax.set_ylim([0, largest_count])

    ax.plot(perclass_performances[:, 0], largest_count * perclass_performances[:, num + 1], c='b', label='Performance')

    ax.set_title(f'MNIST digit {num}')

    if num == 0:
        ax.set_ylabel('Cumulative counts in training data', color='r')

    if num > 0:
        ax.get_yaxis().set_ticklabels([])
        ax.get_xaxis().set_ticklabels([])

    if num == len(axarr_c) - 1:
        ax.legend()
        ax2 = ax.twinx()
        ax2.set_ylabel('Performance', color='b')
        ax2.tick_params('y', colors='b')
plt.suptitle('Compare the cumulative counts and the accuracy per MNIST digit \n '
             'In this experiment, the initial training data only contains MNIST digits below 4\n '
             'Policy name is ' + policy_name)

plt.show()


