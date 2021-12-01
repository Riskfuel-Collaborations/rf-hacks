# Riskfuel Problem Set 


![](https://media-exp1.licdn.com/dms/image/C4D0BAQHa_yrMUj4Fwg/company-logo_200_200/0/1575957122798?e=2159024400&v=beta&t=Te0m8CUYKG3PNIkwZd4rWo1ZwQm0_lAB60hHWA-S6po)


## Introduction
Welcome to MacHacks! My name is Nik and I am representing Riskfuel (RF) as a judge during this hackathon. 


The RF problem set consists of training an ML model to approximate the analytic black scholes pricer of a standard european put option. The ML model will be evaluated on a validation set defined on a closed domain of input parameters. The winner will be chosen based on model performance (ie. best Mean Absolute Error). 


## Black Scholes Model 

The Black Scholes model (BSM) was concieved by a McMaster alumni Myron Scholes and his co-author Fischer Black. From the BSM, one can derive an analytic solution to a European Put option, the solution has been provided to you in `utils/black_scholes.py`. The pricer takes in the following variables: 

- Stock price (S) 
- Strike (K)
- Time to maturity (T)
- Risk free interest rate (r) 
- Volatility (sigma)

Which then outputs the following price: 

- Price of put option in dollars (value) 

Essentially it is a function which takes in 5 inputs, and returns 1 output. 


![](media/bsm.png)


## Domains
The domain on which the model will be validated on. 

```python
S_bound = [0.0, 200.0]
K_bound = [50.0, 150.0] 
T_bound = [0.0, 5.0]
r_bound = [0.001, 0.05]
sigma_bound = [0.05, 1.5]

```

## Constraints/Requirements 

The `riskfuel_test.py` file has validation code on how each team will be evaluated. I must be able to download your git repo, run `pip install -r requirements.txt`, and then run 

```bash 
python riskfuel_test.py --data <validation.csv>
```   

The output should look something like this: 

```python 
 ============ Evaluating Team: Riskfuel ========================= 
 Members :
 Nikola Pocuca
 Maxime Bergeron
 ================================================================ 

 MODEL PERFORMANCE: 45.183143615722656 

```

The validation code will be checked manually for all teams. 

## *Do not put the analytic pricer as part of your model. This is considered cheating and you will be automatically disqualified.* 

You are free to use whatever packages/frameworks you like provided that I am able to install it on a `ubuntu 20.04` docker image found here `https://hub.docker.com/_/ubuntu`. When writing code to evaluate your model, feel free to delete the skeleton code within `riskfuel_test.py`. 

## Recommendations 

Pytorch is a great framework for training ML models that has options for CPU training. Pytorch also allows you to implicity define gradients in an interprative fashion as opposed to other frameworks such as Tensorflow. 

I whole-heartedly advocate Pytorch for this problem set `https://pytorch.org/` over other packages. You will NOT be penalized for not using Pytorch if you are more comfortable with other packages. 


