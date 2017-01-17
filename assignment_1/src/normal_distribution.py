from import_data import *
import scipy.stats as stats
import pylab as pl

#X, attribute_names, dict = import_data()
data = Data()


def plot_feature(index, title, x_axis):
    #chi, p = stats.normaltest()
    #print 'Attribute: ', attribute_names[index], 'chi, p: ', chi, p

    h = sorted(data.X[:, index])

    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed
    pl.plot(h, fit, '-o')
    pl.xlabel(x_axis)
    pl.ylabel('Probability Density (1E-5)')
    pl.title(title)
    pl.hist(h, normed=True)  # use this to draw histogram of your data

    locs, labels = pl.yticks()
    pl.yticks(locs, map(lambda x: "%.1f" % x, locs * 1e5))

    pl.savefig('img/norm_' + data.attribute_names[index] + '.pdf')
    pl.show()


zipped = zip(data.attribute_names, range(len(data.attribute_names)))
print zipped

plot_feature(9, 'GDP per Capita with Normal Distribution', 'GDP per Capita (USD)')
