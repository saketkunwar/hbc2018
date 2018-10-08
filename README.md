# hbc2018
Solution To Human Behavior Challenge 2018
The necessary compition data are in train and test directory.

cd scripts


1. check all requirements with (python2.7 environment)
	pip install -r requirements.txt

2. Run run_train.sh in scripts dir in a python2.7 environment which creates the augmented feature, trains 
   and creates test submission including a cross validation results, stored in 'output' directory.
    ./run_train.sh	

3. Optionally you can run directly using main.py  ..i.e
   python main.py --execute train --indir ../output/ 

4. Retest if required, provided the augmented training data and classifiers have been generated. i.e from step in 3
   python main.py --execute test --indir ../example_run/

5. Retest using augmented training data from step 3 but retrain the classifiers.
   python main.py --execute test --indir ../output/ --retrain True
