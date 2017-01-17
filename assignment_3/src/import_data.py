# exercise 2.1.1
import numpy as np
import xlrd
from similarity import binarize
from pylab import *


class Data:

    def __init__(self, file_name='Country_data_final.xlsx'):
        # Load xls sheet with data
        doc = xlrd.open_workbook(file_name).sheet_by_index(0)

        # Extract attribute names
        attribute_names = doc.row_values(0, 0, doc.ncols)

        # Global variables in method
        cols = doc.ncols
        rows = doc.nrows
        index_of_gov_type = attribute_names.index('gov_type')
        index_of_country = attribute_names.index('country')
        index_of_pop = attribute_names.index('population')
        col_continent_start = 1
        col_continent_end = 7

        # Extract class names in gov_type to python list
        gov_labels = doc.col_values(index_of_gov_type, 1, doc.nrows)
        gov_classes = sorted(set(gov_labels))
        gov_dict = dict(zip(gov_classes, range(len(gov_classes))))

        cont_labels = doc.row_values(0, col_continent_start, col_continent_end)
        cont_classes = sorted(set(cont_labels))
        cont_dict = dict(zip(cont_classes, range(len(cont_classes))))

        for cl in gov_classes:
            attribute_names.append(cl)

        # +1 for the additional column of continent
        total_length = cols + len(gov_classes) + 1
        mat = np.zeros(shape=(rows - 1, total_length))

        # Adding new column
        attribute_names.append('continent')
        index_of_continent = total_length - 1

        for row in range(1, rows):
            for index in range(0, cols):
                row_in_matrix = row - 1
                cell_value = doc.cell_value(row, index)

                # Overwrite country from
                if index == index_of_country:
                    continue

                # continent-interval
                if col_continent_start <= index <= col_continent_end:
                    if cell_value == 1:
                        val = cont_dict[attribute_names[index]]
                        mat[row_in_matrix][index_of_continent] = val

                # If it is the index of the gov type col, take the
                # value, find the matching column and set it = 1
                if index == index_of_gov_type:
                    gov_type_col = attribute_names.index(cell_value)
                    mat[row_in_matrix][gov_type_col] = 1
                    cell_value = gov_dict[cell_value]

                if index == index_of_pop:
                    temp = str(cell_value).replace(',', '').replace('.', '')
                    cell_value = float(temp)

                if cell_value == '':
                    continue

                mat[row_in_matrix][index] = float(cell_value)

        X = np.delete(mat, [index_of_country], 1)
        attribute_names.pop(index_of_country)

        # Preallocate memory, then extract excel data to matrix X

        # Compute values of N, M and C.
        self.X = X
        self.N = rows - 1
        self.M = len(attribute_names)
        self.C_gov = len(gov_classes)
        self.C_cont = len(cont_classes)
        self.gov_classes = gov_classes
        self.cont_classes = cont_classes
        self.attribute_names = attribute_names
        self.y_gov = self._remove_col('gov_type')
        self.y_cont = self._remove_col('continent')

    # Private method, hashtag don't care
    def _remove_col(self, col):
        index = self.attribute_names.index(col)
        y = self.X[:, index]
        self.X = np.delete(self.X, [index], 1)
        self.attribute_names.pop(index)
        self.M -= 1
        return y

    def remove_and_get_y(self, col_name):
        index = self.attribute_names.index(col_name)
        y = self.X[:, index]
        self._remove_col(col_name)
        return y

    # GG
    def remove_one_out_of_k(self, class_names):
        for class_name in class_names:
            index = self.attribute_names.index(class_name)
            self.attribute_names.pop(index)
            self.X = np.delete(self.X, [index], 1)

        self.M = len(self.attribute_names)

    # GG normalisering
    def normalize_data(self):
        self.X = self.X - np.ones((self.N, 1)) * self.X.mean(0)
        self.X = self.X / (np.ones((self.N, 1)) * self.X.std(0))

    def binarize_in_you_face(self):
        x = binarize(self.X)
        self.X[:, 6:9] = x[:, 6:9]

    def get_pca(self):
        # PCA by computing SVD of Y
        U, S, V = linalg.svd(self.X, full_matrices=False)
        V = mat(V).T

        # Project the centered data onto principal component space
        Z = self.X * V
        return Z


