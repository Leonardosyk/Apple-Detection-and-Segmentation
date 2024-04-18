# # 绘制传统和多分类对比
# import matplotlib.pyplot as plt
#
#
#
# data = [0.49, 0.7, 0.8]
# labels = ['CHT', 'HSV', 'multi-classification']
#
# plt.bar(range(len(data)), data, tick_label=labels)
#
# plt.xlabel('Approach')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Define base and platform coordinates (example values, please adjust to your specifics)
base = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])  # Base points PB1, PB2, PB3
platform = np.array([[0.5, 0.2], [0.7, 0.2], [0.6, 0.4]])  # Platform points PP1, PP2, PP3

# Define the center of the platform for annotation
center_of_platform = np.mean(platform, axis=0)

# Plot the base
plt.plot(*zip(*np.append(base, [base[0]], axis=0)), marker='o', color='black')

# Plot the platform
plt.plot(*zip(*np.append(platform, [platform[0]], axis=0)), marker='o', color='blue')

# Plot the arms
for pb, pp in zip(base, platform):
    plt.plot([pb[0], pp[0]], [pb[1], pp[1]], 'k--')

# Annotate the center of the platform
plt.plot(*center_of_platform, 'ro')  # Red dot
plt.annotate('{C}', center_of_platform, textcoords="offset points", xytext=(-10,-10), ha='center')

# Additional annotations and labels can be added similarly using plt.annotate

# Set equal scaling by aspect ratio
plt.axis('equal')

# Turn off the axes
plt.axis('off')

# Show the plot
plt.show()
