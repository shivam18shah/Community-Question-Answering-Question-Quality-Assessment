# CSE_575_SML
Final Project

Sample execution query for terminal:
$ python <model_name>.py
model_name can be log_reg, svm, nn

Python 3 environment
Modules required to be installed: pandas, numpy, matplotlib, pickle, sklearn
Any recent stable version should work.

The data in read from dataset/train.csv, and the outputs are written into Outputs/<filename>. Note that the file must exist for the output to be written, as the program is not creating one. Models and their coefficients are stored in separate pickle files, they can be retrieved by calling pickle.load on the corresponding file.
  

Before running the actual model on the dataset, the program will ask you for an input to make sure that you have securely saved the model from the previous execution, as the file will be overwritten. You can press enter and continue (blank input is fine, just press enter.) is you are ready to execute. Make sure to save the copy of the files after execution, and keep track of the cross validatoin accuracies along with them, as they are NOT savedanywhere.

As of the current commit, nn is not ready to be executed, while log_reg and svm are.
