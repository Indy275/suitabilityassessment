import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap


# Original results
original = np.array([2.5, 2.5, 3.5, 3.5, 2.5])

# Modified row results
modified = np.array([3.17, 3.5, 4.17, 3.5, 3.17])

# Excluded row results
excluded = np.array([3.0, 3.0, 4.0, 3.6, 2.8])

# Column labels
labels = ['Flooding risk of primary embankments',
          'Flooding risk of regional embankments',
          'Ground subsidence',
          'Bottlenecks excessive rainwater',
          'Soil water storage capacity']
labels = ['\n'.join(wrap(x, 20)) for x in labels]

# Plotting
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, original, width, label='Original')
rects2 = ax.bar(x, modified, width, label='Modified')
rects3 = ax.bar(x + width, excluded, width, label='Excluded')


# Customize the plot
ax.set_ylabel('Values')
ax.set_title('Comparison of Results')
ax.set_xticks(ticks=range(0, len(labels)), labels=labels, rotation=90)
ax.set_ylim((0, 4.5))

ax.set_xticklabels(labels)
ax.legend()


# Add value labels to the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Display the plot
plt.tight_layout()
plt.show()

# Plotting
x = np.arange(len(labels))
width = 0.2
plt.plot(x, original, label='Original', marker='o', linestyle='-', linewidth=2)
plt.plot(x, modified, label='Modified', marker='s', linestyle='--', linewidth=2)
plt.plot(x, excluded, label='Excluded', marker='^', linestyle=':', linewidth=2)
# Customize the plot
plt.xlabel('Columns')
plt.ylabel('Values')
plt.title('Comparison of Results')
plt.xticks(x, labels)
plt.legend()

# Add value labels to the data points
def add_labels(data):
    for i, val in enumerate(data):
        plt.text(i, val, str(val), ha='center', va='bottom')

add_labels(original)
add_labels(modified)
add_labels(excluded)

# Display the plot
plt.tight_layout()
plt.show()






