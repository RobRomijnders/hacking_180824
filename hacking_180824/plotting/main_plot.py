from glob import glob
import matplotlib.pyplot as plt
import numpy as np

f, axarr = plt.subplots(2, 2)
colors = ['r', 'b', 'm', 'y', 'k']

for num_file, filename in enumerate(glob('log/*.log')):
    performances = []
    perclass_performances = []
    with open(filename) as f:
        policy_name = 'unknown'
        for line in f:
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

    performances = np.array(performances)
    perclass_performances = np.array(perclass_performances)

    axarr[0, 0].plot(performances[:, 0], performances[:, 1], label=policy_name, c=colors[num_file])
    axarr[0, 1].plot(perclass_performances[:, 0], perclass_performances[:, 1:], c=colors[num_file], label=policy_name)
axarr[0, 0].legend()
axarr[0, 1].legend()
plt.show()


