from read_data import read_data, plot_datapoint
import numpy as np


def main():
    train_inputs, test_inputs, train_labels, test_labels = read_data()

    plot_datapoint(train_inputs[0])
    
    

if __name__ == "__main__":
    main()
