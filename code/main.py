from data_preprocess import *
from time import time
from make_submission import *
from model import *
from add_features_ import *

if __name__ == '__main__':
    start = time()
    ### Data Preprocess  ###
    #data_process_object = Data_Preprocess()
    #matrix, oldcols, train, items, shops = data_process_object()
    #matrix = add_features(matrix, train, items, oldcols, shops)
    #########################
    train_model()
    make_sub()
    end = time()
    print("Execution Time: ", (end - start)/60, " minutes")