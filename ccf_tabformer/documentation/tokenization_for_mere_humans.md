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

## A new pespective on numerical values

The Main kaggle notebook that nudged me into the generative / statistical inference direction revealed a very interesting assumption / property. Numerical values aren't much of numerical quantities but more of generation fingerprints. 

This made me think of the following: 

for each numerical feature, let's first find the maximum number of slots / integers in this column: for example the numerical features is distributed from 0.05 to 2006.34. Thefore, we can view each value as a sequence of 6 digits: 

* 1.45 -> 0001.45
* 234.78 -> 0234.78 

With this representation, We can avoid the headache of finding suitable repreentation of numerical / continous quantities and reduce any number into a squence of low dimensionality tokens. 

A continous number is a sequence of length: digit_slots = int_slots + decimal_slots (int_slots is the maximum number of digits in the integral part and decimal_slots is the maximum number of decimal points)

These widths are measured from the pooled train+test CSV **strings** (not from floats) by `experiments/data/digit_slots.py`. Zero-padding to the column max is lossless: `1.45` with `int_slots=4`, `decimal_slots=2` becomes `0001.45`.

| column | int_slots | decimal_slots | digit_slots | n at max decimal_slots |
|---|---|---|---|---|
| amount_usd | 4 | 3 | 7 | 1 |
| distance_from_home_km | 2 | 2 | 4 | 917680 |
| account_balance_usd | 5 | 2 | 7 | 886974 |
| velocity_score | 2 | 2 | 4 | 1 |
| merchant_risk_score | 3 | 2 | 5 | 1 |

Three of the decimal widths are carried by a **single** written value. `n at max decimal_slots` is how many train+test rows are actually written with that column's maximum number of digits after the decimal point.

| column | written value | split | id | is_fraud |
|---|---|---|---|---|
| amount_usd | 9.982 | test | 962974 | (no label) |
| velocity_score | 29.11 | test | 819176 | (no label) |
| merchant_risk_score | 19.12 | train | 662731 | 0 |

Keeping that extra slot lengthens every row; dropping it discards a one-off generator fingerprint.

### Decision: the last decimal is discarded for these values

Drop the last decimal digit of those three written values (`9.982` → `9.98`, `29.11` → `29.1`, `19.12` → `19.1`). `experiments/data/discard_extra_decimals.py` writes the result to `storage/data/corrected/` (`train.csv`, `test.csv`, `manifest.json`). `storage/raw_data/` stays the generator's original handwriting. All later analysis and training read the corrected copy (`get_train_csv_path` / `get_test_csv_path`).

Widths on the corrected copy, same measurement (`experiments/data/digit_slots.py --source corrected`):

| column | int_slots | decimal_slots | digit_slots | n at max decimal_slots |
|---|---|---|---|---|
| amount_usd | 4 | 2 | 6 | 918777 |
| distance_from_home_km | 2 | 2 | 4 | 917680 |
| account_balance_usd | 5 | 2 | 7 | 886974 |
| velocity_score | 2 | 1 | 3 | 999944 |
| merchant_risk_score | 3 | 1 | 4 | 999944 |

## From row to tokens

Now, let's be specific here. 

The starting point here is

```
"amount_usd": numerical 
"distance_from_home_km": numerical,
"account_balance_usd": numerical,
"velocity_score": numerical,
"merchant_risk_score": numerical,

"hours_since_last_txn": in-between,
"txn_count_last_24h": in-between,
"card_age_months": in-between,
"customer_age": in-between,
"cvv_retry_count": in-between,
"time_of_day_hour": in-between,
"day_of_week": in-between,
"prior_disputes": in-between,

"merchant_category": categorical,
"card_type": categorical,
"auth_method": categorical,
"channel": categorical,
"device_type": categorical,

"is_foreign_transaction": boolean (still categorical),
"is_new_merchant": boolean,
"used_vpn": boolean,
"ip_country_mismatch": boolean,
"billing_shipping_mismatch": boolean,
"is_ai_generated_scam_attempt": boolean: 
```

The tokens that we will work with are: 

1. digits: numerical features

2. either we use digits for in-between features or we treat them as categorical with high cardinality. The high cardinality is only problematic for EDA and mere humands such as myself. Transformers are trained with vocabularies of 50k and more. Let's call the first approach, in_between_as_num and the latter in_between_as_cat 

3. True / False for boolean columns

4. for each categorical, each categorical value is assigned an id.

The mapping below is the lexicographic id of each written value on the pooled train+test working copy, measured by `experiments/data/categorical_values.py`. Every value appears in both splits (no train-only or test-only category).

| column | id | value |
|---|---|---|
| merchant_category | 0 | Crypto Exchange |
| merchant_category | 1 | Electronics |
| merchant_category | 2 | Fuel |
| merchant_category | 3 | Gaming |
| merchant_category | 4 | Gift Cards |
| merchant_category | 5 | Groceries |
| merchant_category | 6 | Healthcare |
| merchant_category | 7 | Online Retail |
| merchant_category | 8 | Restaurants |
| merchant_category | 9 | Streaming |
| merchant_category | 10 | Travel |
| merchant_category | 11 | Utilities |
| card_type | 0 | Amex |
| card_type | 1 | Discover |
| card_type | 2 | Mastercard |
| card_type | 3 | RuPay |
| card_type | 4 | Visa |
| auth_method | 0 | 3D Secure |
| auth_method | 1 | Biometric |
| auth_method | 2 | No Authentication |
| auth_method | 3 | OTP |
| auth_method | 4 | PIN |
| channel | 0 | ATM |
| channel | 1 | Contactless |
| channel | 2 | In-App |
| channel | 3 | Online |
| channel | 4 | POS |
| device_type | 0 | ATM Machine |
| device_type | 1 | Android Phone |
| device_type | 2 | Mac |
| device_type | 3 | POS Terminal |
| device_type | 4 | Smart Watch |
| device_type | 5 | Tablet |
| device_type | 6 | Windows PC |
| device_type | 7 | iPhone |

### IN_between_as_num

We start with **in_between_as_num**: in-between columns are written as digit sequences and share the digit ids. `in_between_as_cat` (each distinct in-between value gets its own id) is the more interesting follow-up; it is not this vocab.

One global vocabulary, laid out in this order:

1. digits first: `0 → 0`, `1 → 1`, …, `9 → 9`
2. then the categoricals, in the feature-list order, each block starting where the previous one ended. `merchant_category` local id `0` (`Crypto Exchange`) becomes global `10`
3. boolean at the end: shared `False` / `True` (the same two ids for every flag column)

The table above is the **local** id (lexicographic, per column). The **global** id is that local id plus the start of the column's block:

| block | n | global ids | first value | last value |
|---|---|---|---|---|
| digits | 10 | 0 – 9 | 0 | 9 |
| merchant_category | 12 | 10 – 21 | Crypto Exchange (10) | Utilities (21) |
| card_type | 5 | 22 – 26 | Amex (22) | Visa (26) |
| auth_method | 5 | 27 – 31 | 3D Secure (27) | PIN (31) |
| channel | 5 | 32 – 36 | ATM (32) | POS (36) |
| device_type | 8 | 37 – 44 | ATM Machine (37) | iPhone (44) |
| boolean | 2 | 45 – 46 | False (45) | True (46) |

So `Visa` is `22 + 4 = 26`, `OTP` is `27 + 3 = 30`, `iPhone` is `37 + 7 = 44`. In-between features do not own a range here; their tokens are digits in `0 – 9`. Vocab size is **47**.

