import matplotlib.pyplot as plt

# Data
suffix_len = [1, 2, 4, 8, 10, 12, 15]
success_rate = [28.33, 25.00, 25.83, 27.50, 24.17, 26.67, 29.17]

# Plot
plt.figure()
plt.plot(suffix_len, success_rate, marker='o')
plt.xlabel('Suffix Length')
plt.ylabel('Success Rate (%)')
plt.xticks(suffix_len)
plt.ylim(min(success_rate) - 1, max(success_rate) + 1)
plt.grid(True)
plt.show()
