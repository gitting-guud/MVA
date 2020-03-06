You need the :
- sequoia-corpus+fct.mrg_strict 
- polyglot-fr.pkl

files to be present in the same repository as the main.py

In the terminal you just need to execute the `run.sh` file without any argument.

This will proceed to : 
- read the data
- split it accoding to the fixed seed
- train on the train+dev dataset
- infer on the test data set (this takes a long time as the CYK is not optimized)
- output a file named `evaluation_data.parser_output` containing the predictions

The libraries needed to run the script are : 
- nltk == 3.4.5
- numpy
- pandas
- sklearn 