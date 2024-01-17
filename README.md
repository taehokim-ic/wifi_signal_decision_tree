# WiFi Signal Decision Tree

### Setting up virtual environment and installing packages

Make sure you have Python 3.8 or above.

#### Install virtual environment
```python3 -m venv venv```
#### Activate virtual environment
```source venv/bin/activate```
#### Install required packages
```pip install -r requirements.txt```

#### TLDR;
Alternatively, the above steps can also be executed using one of the following commands:

Fish Shell\
```. scripts/fish_setup.sh```\
\
Bash Shell \
```. scripts/bash_setup.sh```

### Testing the Datasets

To test the clean dataset directly, run ```python test_clean_dataset.py```.

To test the noisy dataset directly, run ```python test_noisy_dataset.py```

The above commands also plot a visual of the decision tree constructed using the entire dataset supplied.

### Testing with Unseen Dataset
If you have data (.txt) of the following form

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | Label |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| -57 | -63 | -68 | -72 | -52 | -62 | -68 | 4 |
| ... | ... | ... | ... | ... | ... | ... | ... |

Place the dataset  under ```./wifi_db```, and run ```python test_unseen_dataset.py```.\
You will be asked to enter the path of the dataset. Type ```./wifi_db/{name of data file}```.

#### Evaluation

Evaluate - Takes as arguments test_db and trained_tree, and returns the metrics (confusion matrix, accuracy, precision, recall and f1 score) computed by using the test dataset on the trained tree.

kFold_decision_tree_evaluation - Performs 10-fold cross-validation on the dataset. Since no pruning is performed, on each iteration, 9 folds are used for training the model and 1 model is used to test the trained model.
