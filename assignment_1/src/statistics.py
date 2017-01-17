from import_data import *
from pylab import *

# import numpy as np

data = Data()

attribute_names = data.attribute_names
X = data.X

pop_index = attribute_names.index('population')
gdp_index = attribute_names.index('gdp_per_cap')
unemp_index = attribute_names.index('unemployment_rate')

def sum_stat(col):
    # Compute values for unemployment_rate
    mean_x = col.mean()
    std_x = col.std(ddof=1)
    median_x = np.median(col)
    range_x = col.max() - col.min()
    return col, mean_x, std_x, median_x, range_x


def box_class_plot(class_dict, class_col_index, type_index, class_legend, y_axes_title, boxplt_title):

    res = []

    for c in range(class_dict):
        #subplot(1, class_dict, c + 1)
        class_mask = (X[:, class_col_index] == c).ravel()  # binary mask to extract elements of class c
        # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c

        res.append(X[class_mask, type_index])


        # title('Class: {0}'.format(classNames[c]))

    boxplot(res)
    title(boxplt_title)

    xticks(range(1, len(class_legend) + 1), class_legend, rotation=45)
    ylabel(y_axes_title)
    x_type_col = X[:, type_index]
    y_up = x_type_col.max() + (x_type_col.max() - x_type_col.min()) * 0.1;
    y_down = x_type_col.min() - (x_type_col.max() - x_type_col.min()) * 0.1
    ylim(y_down, y_up)

    show()


population, mean_pop, std_pop, median_pop, range_pop = sum_stat(X[:, pop_index].T)
# Display results
print 'population'
print 'Vector:', population
print 'Mean:', mean_pop
print 'Standard Deviation:', std_pop
print 'Median:', median_pop
print 'Range:', range_pop

gdp_per_cap, mean_gdp, std_gdp, median_gdp, range_gdp = sum_stat(X[:, gdp_index].T)
# Display results
print 'gdp_per_cap'
print 'Vector:', gdp_per_cap
print 'Mean:', mean_gdp
print 'Standard Deviation:', std_gdp
print 'Median:', median_gdp
print 'Range:', range_gdp

unemployment_rate, mean_unemp, std_unemp, median_unemp, range_unemp = sum_stat(X[:, unemp_index].T)
# Display results
print 'unemployment_rate'
print 'Vector:', unemployment_rate
print 'Mean:', mean_unemp
print 'Standard Deviation:', std_unemp
print 'Median:', median_unemp
print 'Range:', range_unemp

print attribute_names[7:10], range(1,3)

# figure(figsize=(14,7))
# boxplot(unemployment_rate)
# title('Unemployment rate - boxplot')
# ylabel('Unemployment rate')
# show()
#
# boxplot(gdp_per_cap)
# title('gdp_per_cap - boxplot')
# ylabel('gdp_per_cap')
# show()
#
# boxplot(population)
# title('Population - boxplot')
# ylabel('Population in millions')
#
# show()

gov_type_index = attribute_names.index('gov_type')
cont_type_index = attribute_names.index('continent')
print attribute_names
attr_for_box = [attribute_names[unemp_index],
                attribute_names[gdp_index],
                attribute_names[pop_index]]
print X[:,gov_type_index]
f = figure(figsize=(14,7))


box_class_plot(data.C_gov, gov_type_index, unemp_index, data.gov_classes, 'Unemployment Rate (%)', 'Unemployment Rate')
box_class_plot(data.C_gov, gov_type_index, pop_index, data.gov_classes, 'Population in mill.', 'Population')
box_class_plot(data.C_gov, gov_type_index, gdp_index, data.gov_classes, 'GDP per Capita (USD)', 'GDP per Capita')
box_class_plot(data.C_cont, cont_type_index, unemp_index, data.cont_classes, 'Unemployment Rate (%)', 'Unemployment Rate')
box_class_plot(data.C_cont, cont_type_index, pop_index, data.cont_classes, 'Population in mill.', 'Population')
box_class_plot(data.C_cont, cont_type_index, gdp_index, data.cont_classes, 'GDP per Capita (USD)', 'GDP per Capita')