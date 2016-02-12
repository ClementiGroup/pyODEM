""" Basic example of a run through using the methods in this package"""

##import the basic things
import max_likelihood as ml
import ml.basic_functions as bf
import ml.helper_classes.q_value.q_value as qval

#load a histogram data
hist_data = np.loadtxt("hist.dat") ##assumes number of counts in a bin
edge_data = np.loadtxt("edges.dat") ##assumes equal to the number of 

centers = 0.5 * (edge_data[:-1]+edge_data[1:])

integrated_total = 0.0
for i,v in enumerate(hist_data):
    integrated_total += v*(edge_data[i+1] - edge_data[i])

hist = hist_data / integrated_total
errors = np.sqrt(hist_data) / integrated_total

q_function_list = []

for i in range(len(hist_data)):
    q_function_list.append(bf.statistical.wrapped_gaussian(hist[i], errors[i]))

qfunction = qval(q_function_list)

print qfunction.get_Q(hist)






