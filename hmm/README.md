# Hidden Markov Models

## Definitions

#### Stochastic process

[Wikipedia](https://en.wikipedia.org/wiki/Stochastic_process) ->
Collection of random variables that:
- are indexed by some mathematical set called an *index state*
- take values in a set called *state space*

#### Markov model

Discrete-time stochastic process that verifies two properties:
- **Limited horizon**: the prediction of the state 
![formula](https://render.githubusercontent.com/render/math?math=X_{t})
does not depend upon one or few previous state(s) 
![formula](https://render.githubusercontent.com/render/math?math=X_{t-1}%2C...%2CX_{t-k}).
Such memoryless property for a Markov model of order *k* is defined by:
![formula](https://render.githubusercontent.com/render/math?math=%5Cforall+t%5Cge+0,%5Ctext{P}_{X_t%5Cmid+X_0,X_1,...,X_{t-1}}=%5Ctext{P}_{X_t%5Cmid+X_{t-k},...,X_{t-1}})
- **Stationarity**: the studied model does not evolve with time, so its 
parameters are assumed to be constant. A stochastic process is stationary of
order *k* if and only if 
![formula](https://render.githubusercontent.com/render/math?math=%5Cforall+t,%5Ctext{P}_{X_{t%2B+k}%5Cmid+X_t,...,X_{t%2B+k-1}}=%5Ctext{P}_{X_k%5Cmid+X_0,...,X_{k-1}})

Given the properties of a Markov model (assuming order 1), the joint 
distribution of all 
![formula](https://render.githubusercontent.com/render/math?math=(X_{t})_{t%5Cin%5Cmathbb{N}})
only depends on 
- the distribution
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_0})
of initial state 
![formula](https://render.githubusercontent.com/render/math?math=X_0)
- the distribution of state transitions 
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_{t%2B+1}%5Cmid+X_t}=%5Ctext{P}_{X_1%5Cmid+X_0})

and can be written as:

![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_0,+...,+X_N}(x_0,...,x_N)=%5Ctext{P}_{X_0}(x_0)%5Ctimes+%5Cprod_{i=1}^{N}%5Ctext{P}_{X_1%5Cmid+X_0=x_{i-1}}(x_i))

Learning a Markov model hence consists in inferring
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_0})
and 
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_1%5Cmid+X_0})
from some observations.

#### Markov chain

A Markov chain is a fully observable discrete-state Markov model.

Given *n* possible values for the state, the distribution for the initial state
is specified by a *n*-stochastic vector, and the transition matrix is a 
*n* x *n* stochastic matrix.

As states are fully observable, the Markov Chain can be learnt simply using
the Maximum Likelihood Estimator, by counting:

![formula](https://render.githubusercontent.com/render/math?math=%5Chat{P}_{X_{t%2B+1}=j%5Cmid+X_t=i}=%5Cfrac{N(x_{t%2B+1}=j%5Ccap+x_t=i)}{N(x_t=i)})
where *N* is the number of occurrences.

#### Hidden Markov Models

A Hidden Markov Model is a discrete-state Markov model whose states are 
partially observable. 

Each observation is conditionally independent of all other hidden states and 
all other observations:

![HMM bayesian network](https://upload.wikimedia.org/wikipedia/commons/8/83/Hmm_temporal_bayesian_net.svg)

It is specified by:
- A Markov chain specification (an initial state distribution vector 
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_0})
and a transition distribution matrix
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_{t%2B+1}%5Cmid+X_t})
)
- Emission distributions
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{Y_t%5Cmid+X_t,%5CTheta})
that depend on some parameters 
![formula](https://render.githubusercontent.com/render/math?math=%5CTheta).

If the observations are discrete and can take *m* different values, the emission
distribution of a homogeneous HMM is represented by an *n* x *m* emission matrix
so that the line at index *i* represents the distribution
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{Y_t%5Cmid+X_t=i}).

If the observations are continuous, emission distributions can be represented 
by a list of *n* sets of distribution parameters (for example one mean and 
variance per possible state when all emissions follow normal distributions).


## Applications and algorithms given a HMM

### Probability of an observed sequence

#### Using past and present observations: Bayesian filtering

We are evaluating the state distribution
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_t%5Cmid+Y_0=y_0,...,Y_t=y_t,%5CTheta}).

Let the alpha-coefficients 
![formula](https://render.githubusercontent.com/render/math?math=%5Calpha_t) be
the joint distribution of the current state and the past and current 
observations:

![formula](https://render.githubusercontent.com/render/math?math=%5Calpha_t:x%5Crightarrow+%5Ctext{P}(X_t=x,Y_0=y_0,...,Y_t=y_t))

Given the alpha-coefficients, the state distribution can be computed
straightforwardly by normalization:
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_t%5Cmid+Y_0=y_0,...,Y_t=y_t,%5CTheta}(x)=%5Cfrac{%5Calpha_t(x)}{%5Csum_{x=1}^n+%5Calpha_t(x)})

Alpha-coefficients can be computed recursively using the **forward algorithm**,
well explained by Wikipedia: 
[Forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm#Algorithm):

![formula](https://render.githubusercontent.com/render/math?math=%5Calpha_t(x)=%5Ctext{P}_{Y_t=y_t%5Cmid+X_t=x}%5Ctimes%5Csum_{x'}%5Cleft(%5Ctext{P}_{X_t=x%5Cmid+X_{t-1}=x'}%5Ctimes%5Calpha_{t-1}(x')%5Cright))

#### Using past, present and future observations: Bayesian smoothing

We are evaluating the state distribution
![formula](https://render.githubusercontent.com/render/math?math=%5Ctext{P}_{X_t%5Cmid+Y_0=y_0,...,Y_t=y_t,...,Y_T=y_T,%5CTheta}).

In analogy with alpha-coefficients, let beta-coefficients be defined as the 
likelihood for a state at instant *t* for making the future observations:

![formula](https://render.githubusercontent.com/render/math?math=%5Cbeta_t:x%5Crightarrow+%5Ctext{P}(Y_{t%2B+1}=y_{t%2B+1},...,Y_T=y_T%5Cmid+X_t=x,%5CTheta)) 

Computation of both alpha and beta coefficients are independent and can be done
in parallel. These two computations tasks merged together define the 
[forward-backward algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm#Backward_probabilities).

### Most probable trajectory with the Viterbi algorithm

Given a learned HMM and a sequence of observations, we search the most probable
state trajectory:

![formula](https://render.githubusercontent.com/render/math?math=%5Cleft(x^*_0,...,x^*_T+%5Cright)=%5Cargmax_{x_0,...,x_T}%5Ctext{P}(X_0=x_0,...X_T=x_T%5Cmid+Y_0=y_0,...Y_T=y_t))

Wikipedia: [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)

### Learning a HMM with the Baum-Welch algorithm
