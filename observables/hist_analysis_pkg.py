import numpy as np
import scipy.stats as stats

import math
from pyfexd.observables import Observable


class HistogramO(Observable):
    def __init__(self, nbins, histrange, spacing, edges):
        self.spacing = spacing
        self.histrange = histrange
        self.nbins = nbins
        self.edges = edges

    def compute_observed(self, data, weights):
        if self.edges is None:
            if self.spacing == None:
                hist, edges = np.histogram(data, weights=weights, range=self.histrange, bins=self.nbins)
            else:
                xmin = int(np.min(data)/spacing)
                xmax = int(np.max(data)/spacing) + 1

                hist, edges = np.histogram(x, weights=weights, range=[((xmin*spacing)/(xmax*spacing))], bins=xmax-xmin)
        else:
            hist, edges = np.histogram(data, weights=weights, bins=self.edges)

        normalization = math.fabs(integrate_simple(hist, edges))
        with np.errstate():
            try:
                stdev = np.sqrt(hist) / normalization
                hist = hist / normalization
            except:
                print normalization
                print hist
                print np.sqrt(hist)
        stdev = np.sqrt(hist) / normalization
        hist = hist / normalization
        bincenters = 0.5*(edges[1:] + edges[:-1])

        seen = [not i==0 for i in hist]


        return hist, stdev, seen


def histogram_distance(data, nbins=10, histrange=(0,10), spacing=None, edges=None, weights=None):
    """ Formats the output from histogram_data

    histogram_distance will output things as a hist and stdev.
    histogram_data does all the heavy lifting. As the user can
    note, histogrma_dat outputs everything while this will
    output only the necessary variables for the max_likelihood
    method.

    """
    if spacing == None:
        hist, stdev, bincenters, edges, slices = histogram_data(data, nbins=nbins, histrange=histrange, edges=edges, weights=weights)
    else:
        hist, stdev, bincenters, edges, slices = histogram_data_spacing(data, spacing=spacing, weights=weights)
    return hist, stdev


def histogram_data(data, nbins=10, histrange=(0,10), edges=None, weights=None):
    """ Histogram a 1-d Array, integral normalized

    Histogram is integral normalized. A stdev is outputted as
    sqrt(N)/norm where N is the total counts of each bin and
    norm is the normalization factor the whole histogram is
    divided by.

    Must specify range and number of bins
    Or a list of edges to histogram inside

    Weights is optional and is assumed equal weighting if None

    """

    if weights is None:
        weights = np.ones(np.shape(data)[0])

    if edges is None:
        hist, edges, slices = stats.binned_statistic(data, weights, statistic="sum", histrange=[histrange], bins=nbins)
    else:
        hist, edges, slices = stats.binned_statistic(data, weights, statistic="sum", edges=edges)

    normalization = math.abs(integrate_simple(hist, edges))

    stdev = np.sqrt(hist) / normalization
    hist = hist / normalization
    bincenters = 0.5*(edges[1:] + edges[:-1])

    return hist, stdev, bincenters, edges, slices

def histogram_data_spacing(data, spacing, weights=None):
    if weights is None:
        weights = np.ones(np.shape(data)[0])

    xmin = int(np.min(data)/spacing)
    xmax = int(np.max(data)/spacing) + 1

    hist, edges, slices = stats.binned_statistic(x, weights, statistic="sum", histrange=[((xmin*spacing)/(xmax*spacing))], bins=xmax-xmin)

    normalization = math.abs(integrate_simple(hist, edges))

    stdev = np.sqrt(hist) / normalization
    hist = hist / normalization
    bincenters = 0.5*(edges[1:] + edges[:-1])

    return hist, stdev, bincenters, edges, slices



def integrate_simple(hist, edges):
    sum = 0.0
    for i, val in enumerate(hist):
        sum += val*(edges[i+1] - edges[i])

    return sum
