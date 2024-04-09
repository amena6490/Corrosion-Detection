import torch
from metric_evaluation import plot_confusion_matrix, iterate_data, spectrum_score_norm, spectrum_score


data_dir = './PATH TO DIR/'
batchsize = 1

model = torch.load(f'./stored_weights/weights.pt', map_location=torch.device('cuda'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()   # Set model to evaluate mode
##############################################################################

iOU, f1, confm_sum, y_pred = iterate_data(model, data_dir)

print('iOU: ' + str(iOU))
print('f1 score: ' + str(f1))

plot_confusion_matrix(confm_sum, target_names=['Background', 'Low', 'Moderate', 'Severe'], normalize=True, 
                      title='Confusion Matrix')
