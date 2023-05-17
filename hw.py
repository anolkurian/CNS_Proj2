# -*- coding: utf-8 -*-
""" CIS6261TML -- Homework 2 -- hw.py

# This file is the main homework file
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt


import os
## os / paths
def ensure_exists(dir_fp):
    if not os.path.exists(dir_fp):
        os.makedirs(dir_fp)

## parsing / string conversion to int / float
def is_int(s):
    try:
        z = int(s)
        return z
    except ValueError:
        return None


def is_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return None


import scipy.stats as stats

## noise distributions
"""
## returns sample from the Laplace distribution with mean 0 and shape b > 0
"""
def laplace_noise(b):
    return stats.laplace.rvs(loc=0, scale=b)


"""
## returns sample from the Gaussian distribution with mean 0 and std sigma > 0
"""
def gaussian_noise(sigma):
    return stats.norm.rvs(loc=0, scale=sigma)

"""
## Load the dataset
"""
def load_data(fp = './data/ds.csv'):
    ds = np.loadtxt(fp, delimiter=',', skiprows=0, dtype=str)
    assert ds is not None, 'Could not load dataset {}!'.format(fp)
    
    header = ds[0,:]
    print('Loaded dataset (rows: {})'.format(ds.shape[0]))
    print('Header: {}'.format(header))
    
    return ds[1:], header

"""
## queries; note that the queries also return their (global) sensitivity
"""
def mean_gpa_query(ds, header):
    assert ds is not None
    gpa_idx = 1
    assert header[gpa_idx] == 'gpa'
    
    ## TODO ##
    ## Insert your code here to set the global sensitivity
    sensitivity = 3.0 / len(ds)
    # get as a float as the rest of the dataset is string
    gpas = ds[:,gpa_idx].astype(float) 
    
    return np.mean(gpas), sensitivity


"""
## dp mechanisms
"""
def laplace_mech(ds, query_fn,  epsilon):
    assert epsilon > 0, 'Invalid parameters.'

    answer, sensitivity = query_fn(ds)
    assert sensitivity > 0

    ## TODO ##
    ## Insert your code here (hint: use laplace_noise())
    scale = sensitivity/epsilon
    noisy_answer = answer + laplace_noise(scale)

    return noisy_answer, epsilon


def gaussian_mech(ds, query_fn, epsilon, delta):
    assert epsilon > 0 and 0 < delta < 1.0, 'Invalid parameters.'

    answer, sensitivity = query_fn(ds)
    assert sensitivity > 0

    ## TODO ##
    ## Insert your code here (hint: use gaussian_noise())
    variance = (sensitivity ** 2) / (2 * epsilon ** 2 * np.log(1.25 / delta))
    noise = gaussian_noise(np.sqrt(variance))
    noisy_answer = answer + noise
    return noisy_answer, epsilon


def qual_score_most_popular_course(courses, ds, r):
    assert r in courses
    
    ## TODO ##
    ## Insert your code here
    ## Make sure your implementation is consistent with 'delta_qual_score' you set in main()
    course_idx = courses.index(r)
    count = np.sum(ds == r)
    return count


def exp_mech(ds, outcomes, qual_score_fn, delta_qual_score, epsilon):
    assert epsilon > 0, 'Invalid parameters.'
    assert qual_score_fn is not None and 0 < delta_qual_score, 'Invalid parameters.'

    ## TODO ##
    ## Insert your code here
    ## Important: make sure you implement the mechanism so it satisfies 'epsilon'-DP.

    n = len(outcomes)
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = np.exp(epsilon * qual_score_fn(ds, outcomes[i]) / (2 * delta_qual_score))
    weights /= np.sum(weights)  # normalize weights

    idx = np.random.choice(range(n), p=weights)
    r = outcomes[idx]
    return r, epsilon



def truncate_gpa(x):
    ## TODO ##
    ## Insert your code here
    if x < 1.0:
        return 1.0
    elif x > 4.0:
        return 4.0
    else:
        return x


"""
## plots
"""
def plot_distribution(courses, true_dist, em_fn, epsilon, num_samples=1000, fname='dist_plot.png'):

    # will be an array of num_samples simulations of the exp mech
    em_dist = np.zeros((len(courses),))   
    
    # create histograms out of the simulations array
    for i in range(0, num_samples):
        c, _ = em_fn()
        cidx = courses.index(c)
        em_dist[cidx] += 1.0
    em_dist /= np.sum(em_dist)    
            

    x = np.arange(len(courses))
    xticks = courses

    # plot the stuff
    fig = plt.figure(figsize=(12,7))
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'font.size': 14})
    plt.bar(x, true_dist, color='b', alpha=0.5, label='True', width=0.6)
    plt.bar(x, em_dist, color='r', alpha=0.5, label='Exp Mech (eps={:.2f})'.format(epsilon), width=0.6)

    plt.grid(axis='y', alpha=0.7)
    plt.xlabel('Course')
    plt.ylabel('Probability')
    plt.xticks(ticks=x, labels=xticks, rotation=90)
    ymax = np.minimum(1.0, 1.1*np.amax(np.maximum(true_dist, em_dist)))
    plt.ylim([0, ymax])
    fig.subplots_adjust(bottom=0.25) # ensure we have enough space for xticks
    plt.legend()

    # save the plot
    ensure_exists('./plots')
    out_fp = './plots/{}'.format(fname)
    plt.savefig(out_fp)

    plt.show()      # comment out if the code is invoked in non-interactive contexts


    

def main():


    # figure out the problem number and subproblem number
    assert len(sys.argv) >= 2, 'Incorrect number of arguments!'
    p_split = sys.argv[1].split('problem')
    assert len(p_split) == 2 and p_split[0] == '', 'Invalid argument {}.'.format(sys.argv[1])
    problem_str = p_split[1]

    assert is_number(problem_str) is not None, 'Invalid argument {}.'.format(sys.argv[1])
    problem = float(problem_str)
    probno = int(problem)

    if probno != 3:
        assert False, 'Problem {} is not a valid problem # for this assignment/homework!'.format(problem)

    sp = problem_str.split('.')
    assert len(sp) == 2 and sp[1] != '', 'Invalid problem numbering.'
    subprob = int(sp[1])
    if subprob <= 0 or subprob > 4:
        assert False, 'Problem {} is not a valid problem # for this assignment/homework!'.format(problem)

    data_fp = os.path.join(os.getcwd(), 'data')
    assert os.path.exists(data_fp), 'Can''t find data!'

    # load the dataset
    ds, header = load_data()

    # parameter for all subproblems except 3
    delta = np.power(2.0, -30.0)

    if subprob == 1: ## problem 3.1
        assert len(sys.argv) == 2 or len(sys.argv) == 3, 'Invalid extra argument'
        
        # use epsilon = 1.0 if not specified
        epsilon = 1.0
        if len(sys.argv) == 3:
            epsilon = is_number(sys.argv[2])
            assert epsilon is not None and epsilon > 0

        query_fn = lambda x: mean_gpa_query(x, header)

        true_mean_age, sensitivity = query_fn(ds)
        laplace_answer, eps = laplace_mech(ds, query_fn, epsilon)
        assert eps == epsilon
        laplace_answer = truncate_gpa(laplace_answer)
        
        gaussian_answer, eps = gaussian_mech(ds, query_fn, epsilon, delta)
        assert eps == epsilon
        gaussian_answer = truncate_gpa(gaussian_answer)
        
        print('Problem 3.1: true mean gpa {:.2f}, laplace noisy answer: {:.2f}, gaussian noisy answer: {:.2f} [epsilon = {:.3f}, log2 delta = {}, sensitivity = {:.3f}]'.
                    format(true_mean_age, laplace_answer, gaussian_answer, epsilon, np.log2(delta), sensitivity))

    elif subprob == 2: ## problem 3.2
        assert len(sys.argv) > 2, 'Epsilon value not specified!'
        epsilon = is_number(sys.argv[2])
        assert epsilon is not None and epsilon > 0, 'Invalid epsilon.'
        
        # hardcoded courses list
        courses = ['CAP6516', 'CAP6610', 'CAP6617', 'CAP6701', 'CDA5636', 'CEN5035', 'CEN5726',
 'CEN5728', 'CEN6075', 'CIS5370', 'CIS5371', 'CIS6261', 'CNT5106C', 'CNT5410',
 'CNT5517', 'COP5536', 'COP5556', 'COP5615', 'COP5618', 'COP5725', 'COT5405',
 'COT5441', 'COT5442', 'COT5615']
        
        ## Insert your code here to set the global sensitivity of your quality score function ('delta_qual_score')
        delta_qual_score = 1 ### fill in here
        assert delta_qual_score is not None and delta_qual_score > 0
        
        qual_score_fn = lambda d, r: qual_score_most_popular_course(courses, d, r)
        
        courses_idx = [2,3,4]
        ds_courses = ds[:, courses_idx]
        
        # lambda for invoking the exponential mechanism
        em_fn = lambda: exp_mech(ds_courses, courses, qual_score_fn, delta_qual_score, epsilon)

        true_dist = np.zeros((len(courses),))
        for i, c in enumerate(ds_courses.ravel()):
            true_dist[courses.index(c)] += 1
        true_dist /= np.sum(true_dist)
            
        plot_distribution(courses, true_dist, em_fn, epsilon, num_samples=1000, fname='dist_plot.png')

if __name__ == '__main__':
    main()
