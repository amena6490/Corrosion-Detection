Required libraries:

Pytorch >= 1.4
Python >= 3.6
tqdm
matplotlib
sklearn
cv2
Pillow
pandas
shutil


For training:

1. Change the directory to training_evaluating folder
2. Extract the dataset from the zip folder.
3. Run the following command:

python main_plus.py -data_directory '/DATA DIRECTORY PATH/' -exp_directory '/CHECKPOINTS SAVE DIRECTORY PATH/' \
--epochs 200 --batch 16

If you get out of memory error, please adjust the batch size to 2. 




For Evaluating the model:

1. Change the directory to training_evaluating folder

2. Run the following command:

python run_metrics_evaluation.py




For Visualising the results:

1. Change the directory to visualising folder
2. Run the following command:

python run_show_results__.py 



For preprocessing:

For preprocessing visit the data_preprocessing folder.


