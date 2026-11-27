# Context and thoughts
The competition rewards reverse-engineering the generator's fingerprints. It is not about modeling fraud, but more about finding the marks the model left behind. 

Pretraining sounds like a great direction for such use-cases. 

There is a myriad of choices to make here. So Breaking down the problem into very clear and well-defined steps is key to building a powerful encoder. 


The starting point here is a row in a table. The table contains a fixed number of features. 

Features can be: 

1. categorical: takes a set of predefined values
2. numerical: continuous, no predefined set of possible values
3. in-between: numerical yet with a limited number of possible values (but the limiter number is usually too large for this feature to be seen as categorical, think 60, 70, 100, 150 ...)

Models works with numbers. So we need to convert all the features into numerical representations. The general methodology is the following

1. value -> id: tokenization
2. id -> embedding

and the embedding is learnable. 


# Tokenization

The Main kaggle notebook that nudged me into the generative / statistical inference direction revealed a very interesting assumption / property. Numerical values aren't much of numerical quantities but more of generation fingerprints. 

This made me think of the following: 

for each numerical feature, let's first find the maximum number of slots / integers in this column: for example the numerical features is distributed from 0.05 to 2006.34. Thefore, we can view each value as a sequence of 6 digits: 

* 1.45 -> 0001.45
* 234.78 -> 0234.78 

With this representation, We can avoid the headache of finding suitable repreentation of numerical / continous quantities and reduce any number into a squence of low dimensionality tokens. 