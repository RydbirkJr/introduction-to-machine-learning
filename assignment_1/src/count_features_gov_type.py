from import_data import *
import numpy as np

data = Data()


def print_counts(mat_index, labels):
    data_sum = data.X[:, mat_index].astype('int32')
    y = np.bincount(data_sum)
    ii = np.nonzero(y)[0]

    zipped = zip(ii, y)

    res = map(lambda (index, count): (labels[index], count), zipped)

    print 'Attribute: ', data.attribute_names[mat_index]

    for (country, count) in res:
        print country, ':\t', count

    print ''


cont_index = data.attribute_names.index('continent')
gov_index = data.attribute_names.index('gov_type')
cont_names = data.cont_classes
gov_types = data.gov_classes

print_counts(cont_index, cont_names)
print_counts(gov_index, gov_types)






