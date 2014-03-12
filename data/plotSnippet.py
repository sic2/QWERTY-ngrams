#a = 2DCM Data
#b = SCM Data
p1, =plt.plot(range(len(a)), a.values(), 'ro', color='r')
p2, =plt.plot(range(len(b)), b.values(), 'ro', color='b')
plt.xticks(range(len(a)), a.keys())
plt.legend([p1, p2], ['2DCM', 'SCM'], loc=4)
plt.title('Correcting Any Letter')
plt.show()