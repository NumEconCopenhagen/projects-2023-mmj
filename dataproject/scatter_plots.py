import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
def plot_scatter(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, color='blue', label=ylabel)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trendline = intercept + slope * x
    plt.plot(x, trendline, color='red', label="Trendline")
    for i, stock in enumerate(merged_data['stock']):
        plt.annotate(stock, (x[i], y[i]), textcoords="offset points", xytext=(35, 0), ha='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    print('The slope of the trendline is: {:.4f}'.format(slope))