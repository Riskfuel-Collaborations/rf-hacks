import pandas as pd
import argparse 
import sys

# Load whatever imports you need, but make sure to add them to the requirements.txt file. 


# my imports. 
import numpy as np 
import torch 
import torch.nn.functional as F 
import torch.utils.data as Data 
import torch.nn as nn 

import pandas as pd
import matplotlib.pyplot as plt 


def riskfuel_test(df: pd.DataFrame, ) -> float: 
    """
        
        Riskfuel Testing Function
        by <team-name>: <member_1> <member_2> .... <member_k> 

        arguments: pandas DataFrame type with the following columns.. ['S','K','T','r','sigma','value'] all are of type float32
        ouputs: mean absolute error (float)


        Once you have finished model training/developemnt you must save the model within the repo and load it in using this function.

        You are free to import any python packages you desire but you must add them to the requirements.txt file. 

        This function must do the following: 
            - Successfully load your own model. 
            - Take in a dataframe consisting of (N x 6) float32's. 
            - Take the (N x 5) columns regarding the inputs to the pricer ['S','K','T','r','sigma'] and have your model price them. 
            - Return the Mean  Absolute Error of the model.

        Do not put the analytic pricer as part of your network. 
        Do not do any trickery with column switching as part of your answer. 
        
        These will be checked by hand, any gaslighting will result in automatic disqualification. 


        The following example has been made available to you. 

    """


    # TEAM DEFINITIONS. 
    team_name = "Riskfuel" # adjust this
    members = ['Nikola Pocuca', 'Maxime Bergeron'] # adjust these

    print(f"\n\n ============ Evaluating Team: {team_name} ========================= ")
    print(" Members :")
    for member in members:
        print(f" {member}")
    print(" ================================================================ \n")

    # =====================================================   SKELETON CODE FEEL FREE TO DELETE  =======================================================================================

    # My model uses pytorch but you can use whatever package you like as long you write code to load it and effectively calculate the mean absolute aggregate error. 

    # LOAD MODEL 
    mm = PutNet()
    mm.load_state_dict(torch.load("simple-model.pt"))
    mm.eval() # evaluation mode

    # EVALUATE MODEL

    # acquire inputs/outputs
    X = torch.Tensor(df[['S','K','T','r','sigma']].to_numpy())
    y = torch.Tensor(df[['value']].to_numpy())
    
    # pass data through model. 
    y_hat = mm(X)

    # =====================================================   END OF SKELETON CODE  =====================================================================================================

    # calculate mean squared error.
    result = (y_hat - y).pow(2).mean()

    # return performance metric. 
    return float(result)


# A SIMPLE MODEL. 
class PutNet(nn.Module):
    """ 

    Example of a Neural Network that could be trained price a put option.  
    
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()
        self.l1 = nn.Linear(5,20)
        self.l2 = nn.Linear(20,20)
        self.l3 = nn.Linear(20,20)
        self.out = nn.Linear(20,1)


    def forward(self, x): 
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.out(x)
        return (x)


def get_parsed_args():
    """ Parses the command line for the dataframe file name """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    return parser



def main(args):
    """ MAIN FUNCTION 

    - parses arguments and evaluates model performance

    """
    # Parse arguments. 
    parser = get_parsed_args()

    args = parser.parse_args(args)

    data_frame_name = args.data

    # grab dataframe and pass through riskfuel_test function. 
    df = pd.read_csv(data_frame_name)
    performance_metric = riskfuel_test(df)

    # must pass this test. 
    assert type(performance_metric) == float

    print(f" MODEL PERFORMANCE: {performance_metric} \n\n")



# MAIN CALL. 
if __name__ == "__main__":
    main(sys.argv[1:])
