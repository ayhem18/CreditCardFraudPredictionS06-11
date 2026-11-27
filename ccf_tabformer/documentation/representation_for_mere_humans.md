# Overview

The [tokenization_doc](tokenization_for_mere_humans.md) explained the overall context and the first step of transformer-based modeling. 

This document will go through the representation step: embedding, model and training. 

# Embeddings & stuff

## IN_BETWEEN_AS_NUMERICAL

Transformers expects input of the shape $(B, S, D)$ where

* B: batch size
* S: sequence length
* D: embedding dimension

Well, batch size is clear (I hope). Given how represented each numerical quantity as a sequence with predefined length, each row now has a predetermined length. Each numerical feature is associated with a given length equal to $\text{digit\_slots}$ and each categorical feature takes a single id contributing exactly one unit to the sequence length

$$
\text{Length = sum of categorical / boolean features + sum of (digit\_slots of a given numerical feature)}
$$

Now, to the $M$ part. Each token should be mapped to a numerical vector. The numerical vector is then modified by the model and backpropagation. 

There are a few interesting things happening here: 

1. certain elements of the sequence, although separate, represent the same feature / column. For example, `amount_usd` might be assigned the value $23.56$. The latter is converted to `0023.56`. this is not a signal element in the sequence. It is 6 elements converted to token ids `002356`. We need to explicitly tell the model that these separate elements represent the same feature

2. This takes me to another important point: a zero at the first integer position of the number probably should not be interpreted the same way as a zero in the first / second decimal slot. So we should incorporte the "slot" position information. 


As I am writing this, I realize that tokenization and embeddings are completely separable and are both coupled together. There are a few ways to incorporate the semantics and concers I have just highlighted above: 

### 3 embedding tables: final embedding as a sum

Given the global ids defined in the tokenization step, we should have an embedding mapping an id to a numerical vector

$$ 
\text{token\_embedding: ~ global\_id} \rightarrow \text{token\_initial\_representation}
$$ 

We should also have a general vector that adds the "feature-context"

$$ 
\text{feature\_embedding: ~ feature} \rightarrow \text{feature\_specific\_representation}
$$ 

We should also add the context of the position of the digit within the numerical quantity: 

$$ 
\text{digit\_slot\_embedding: ~ digit slot} \rightarrow \text{digit\_slot\_representation}
$$ 


Then the final representation of a given element is defined as

* $ \text{token\_initial\_representation} + \text{feature\_specific\_representation} $ if the feature is categorical

* $ \text{token\_initial\_representation} + \text{feature\_specific\_representation} +  \text{digit\_slot\_representation} $ if the feature is numerical 


My intuition here tells me that this specific tokenization + embedding strategy lies on an important assumption: 

$$
\text{
the numerical digits are treated by the data generator as actual numerical representation that belong to the actual decimal / numerical system. 
}
$$

More specifically, "9" here actual means the number 9 regardless of the column / feature with which the value is associated. For this specific dataset, it might not be the case. 

Hence, it might be fruitful to experiment with a different strategy  where

1. each digit is associated with a different id depending on the column, which removes the need for a column feature since the id implies the feature
2. a slot position embedding table that associates a different embedding for a given slot position depending on the column. The strategy is explained in more details below

### 2 embedding tables: final embedding as a sum 

