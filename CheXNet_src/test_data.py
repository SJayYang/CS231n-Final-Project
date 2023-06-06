import os

generated_images_path = '/home/ubuntu/CS231n-Final-Project/generated_images'
file_path = '/home/ubuntu/CS231n-Final-Project/CheXNet_src/generated_images.txt'
ACTUAL_TRAIN_LIST = '/home/ubuntu/CheXNet/ChestX-ray14/labels/train_list.txt'
diseases = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
disease_images = []
final_file = '/home/ubuntu/CS231n-Final-Project/CheXNet_src/mixed_train.txt'

for filename in os.listdir(generated_images_path):
    if os.path.isfile(os.path.join(generated_images_path, filename)):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            disease_images.append(filename)

with open(file_path, 'w') as file:
    for disease in disease_images:
        condition = disease.split('_')

        if len(condition) == 2:
            condition = condition[0]
        else:
            condition = condition[:len(condition) - 1]
            condition = '_'.join(condition)

        one_hot = ['0'] * 14
        index = diseases.index(condition)
        one_hot[index] = '1'
        string = disease + ' ' + ' '.join(one_hot)
        file.write(string + '\n')

with open(file_path, 'r') as f1, open(ACTUAL_TRAIN_LIST, 'r') as f2, open(final_file, 'w') as f_out:
        contents_first = f1.read()
        contents_sec = f2.read()
        f_out.write(contents_first + '\n')
        f_out.write(contents_sec)
