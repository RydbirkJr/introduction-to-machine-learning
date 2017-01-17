from import_data import *
from pylab import *
import similarity as sim

# This script plots features with class legend
data = Data()

X = data.X
attribute_names = data.attribute_names


# Data attributes to be plotted
index_ER = attribute_names.index('unemployment_rate')
index_GDP = attribute_names.index('gdp_per_cap')
index_POP = attribute_names.index('population')
index_cont = attribute_names.index('continent')
index_gov = attribute_names.index('gov_type')
gov = data.gov_classes
cont = data.cont_classes


# Plots for both classes
def plot_with_mask2(plot_title, y_axis, x_axis, i, j):
    plot_with_mask(plot_title, y_axis, x_axis, i, j, gov, index_gov)
    plot_with_mask(plot_title, y_axis, x_axis, i, j, cont, index_cont)


# Plots only the given class
def plot_with_mask(plot_title, y_axis, x_axis, j, i, plot_legend, mask_index):

    # Convert to array
    Xa = array(X)

    # %%
    # Make another more fancy plot that includes legend, class labels,
    # attribute names, and a title.
    f = figure()
    f.hold()
    title(plot_title)

    print ''
    for c in range(len(plot_legend)):
        # select indices belonging to class c:
        class_mask = X[:, mask_index].ravel() == c
        x = Xa[class_mask, i]
        y = Xa[class_mask, j]

        print plot_legend[c]
        print 'Correlation of: ', x_axis, y_axis
        print sim.similarity(x, y, 'cor')

        plot(Xa[class_mask, i], Xa[class_mask, j], 'o')

    print 'Total correlation: '
    print sim.similarity(X[:, i], X[:, j], 'cor')

    legend(plot_legend)
    xlabel(x_axis)
    ylabel(y_axis)

    # Output result to screen
    show()


# Plots only the given class
def plot_with_mask_highlight_europe():

    # Convert to array
    Xa = array(X)

    # %%
    # Make another more fancy plot that includes legend, class labels,
    # attribute names, and a title.
    f = figure()
    f.hold()
    title('GDP per Capita vs Unemployment Rate')

    i = index_GDP
    j = index_ER

    c = cont.index('Europe')
    # select indices belonging to class c:
    class_mask = X[:, index_cont].ravel() == c
    x = Xa[class_mask, i]
    y = Xa[class_mask, j]

    print sim.similarity(x, y, 'cor')
    plot(Xa[class_mask, i], Xa[class_mask, j], 'o')

    class_mask = X[:, index_cont].ravel() != c
    x = Xa[class_mask, i]
    y = Xa[class_mask, j]

    print sim.similarity(x, y, 'cor')
    plot(Xa[class_mask, i], Xa[class_mask, j], 'o')

    legend(['Europe', 'Rest'])
    ylabel('Unemployment Rate (%)')
    xlabel('GDP per Capita (USD)')
    savefig('img/cor_europe_rest.pdf')
    # Output result to screen
    show()


def plot_with_mask_highlight_northamerica_australia():

    # Convert to array
    Xa = array(X)

    # %%
    # Make another more fancy plot that includes legend, class labels,
    # attribute names, and a title.
    f = figure()
    f.hold()
    title('GDP per Capita vs Population')

    i = index_GDP
    j = index_POP

    na = cont.index('North America')
    aus = cont.index('Australia')
    # select indices belonging to class c:


    # mask North America
    class_mask_na = X[:, index_cont].ravel() == na
    x = Xa[class_mask_na, i]
    y = Xa[class_mask_na, j]

    print sim.similarity(x, y, 'cor')


    # mask Australia
    class_mask_aus = X[:, index_cont].ravel() == aus
    x = Xa[class_mask_aus, i]
    y = Xa[class_mask_aus, j]

    print sim.similarity(x, y, 'cor')
    # print class_mask_na
    # print class_mask_aus
    # print (class_mask_na + class_mask_aus)
    # print np.invert(class_mask_na + class_mask_aus)

    # Mask rest


    class_mask_rest = np.invert(class_mask_na + class_mask_aus)
    x = Xa[class_mask_rest, i]
    y = Xa[class_mask_rest, j]

    print sim.similarity(x, y, 'cor')
    plot(Xa[class_mask_na, i], Xa[class_mask_na, j], 'o', color='blue', zorder=3)
    plot(Xa[class_mask_aus, i], Xa[class_mask_aus, j], 'o', color='red', zorder=2)
    plot(Xa[class_mask_rest, i], Xa[class_mask_rest, j], 'o', color='white', zorder=1)

    legend(['North America', 'Australia', 'Rest'], loc='upper right')

    xlabel('GDP per Capita (USD)')
    locs, labels = yticks()
    yticks(locs, map(lambda x: "%.0f" % (x / 1e6), locs))
    ylabel('Population (millions)')
    savefig('img/cor_na_aus_rest.pdf')
    # Output result to screen
    show()


#plot_with_mask_highlight_europe()

plot_with_mask_highlight_northamerica_australia()
#
# plot_with_mask2('GDP per Capita vs Unemployment Rate',
#                 'GDP per Capita (USD)',
#                 'Unemployment Rate (%)',
#                 index_GDP,
#                 index_ER)
#
#
# plot_with_mask2('GDP per Capita vs Population',
#                 'GDP per Capita (USD)',
#                 'Population',
#                 index_GDP,
#                 index_POP)
#
# plot_with_mask2('Unemployment Rate vs Population',
#                 'Unemployment Rate (%)',
#                 'Population',
#                 index_ER,
#                 index_POP)

