from import_data import *
from pylab import *

data = Data()


population_index = data.attribute_names.index('population')

h = sorted(data.X[:, population_index])

#fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed
plt.plot(h, 'o')
plt.title('Population')
plt.ylabel('Population size')
plt.xlabel('Country (in sorted order)')
#pl.hist(h, normed=True)  # use this to draw histogram of your data

locs, labels = yticks()
yticks(locs, map(lambda x: "%.0f" % x, locs / 1e6))
ylabel('Population (millions)')

plt.savefig('img/pop_outlier.pdf')

plt.show()
