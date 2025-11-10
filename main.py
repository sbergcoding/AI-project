import sys
import numpy as np
import pandas as pd
import kaggle
import os 

kaggle.api.authenticate()

kaggle.api.dataset_download_files('ananthu017/emotion-detection-fer', path='.',unzip=True)
kaggle.api.dataset_metadata('ananthu017/emotion-detection-fer', path='.')
#print(sys.path)

def main(): 

    print("hello world")

    

if __name__=="__main__": #this is very important 
    main()
    #I guess this is how we are going to start.
    #use python -m venv myproject-env for compiling code
    # then env\Scripts\activate to run
    #to kill env type deactivate
    #print("hello world")
    