from glob import glob
import matplotlib.pyplot as plt
import numpy as np

f, axarr = plt.subplots(2, 2)

for num_file, filename in enumerate(glob('log/*.log')):
    performances = []
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

    performances = np.array(performances)

    axarr[0, 0].plot(performances[:, 0], performances[:, 1], label=policy_name)
axarr[0, 0].legend()
plt.show()


