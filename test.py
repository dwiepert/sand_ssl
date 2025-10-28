from sand_ssl.io import CustomDataset, collate_wrapper
from sand_ssl.models import Extractor, Model
from torch.utils.data import DataLoader
from sand_ssl.training import Trainer


epochs = 1
debug = True

audio_dir = '/Users/dwiepert/Documents/GITHUB/sand_ssl/task1_train/training'
spreadsheet_path = '/Users/dwiepert/Documents/GITHUB/sand_ssl/task1_train/sand_task_1.xlsx'
metadata_path = '/Users/dwiepert/Documents/GITHUB/sand_ssl/task1_train/task1_metadata.csv'

audio_dir2 = '/Users/dwiepert/Documents/GITHUB/sand_ssl/task1_test/test'
spreadsheet_path2 = '/Users/dwiepert/Documents/GITHUB/sand_ssl/task1_test/sand_task1_test.xlsx'
model_type='wavlm-base'
batch_sz = 2

training_data = CustomDataset(audio_dir=audio_dir, spreadsheet_path=spreadsheet_path, metadata_path=metadata_path, data_type='training', debug=debug)
validation_data = CustomDataset(audio_dir=audio_dir, spreadsheet_path=spreadsheet_path, metadata_path=metadata_path, data_type='validation', debug=debug)
#testing_data = CustomDataset(audio_dir=audio_dir2, spreadsheet_path=spreadsheet_path2, debug=False)

feature_extractor = Extractor(model_type, False)
collate_fn = collate_wrapper(feature_extractor)
train_loader = DataLoader(dataset=training_data, batch_size=batch_sz, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(dataset=validation_data,batch_size=batch_sz, shuffle=False, collate_fn=collate_fn, num_workers=0 )
#test_loader = DataLoader(dataset=testing_data, batch_size=batch_sz, shuffle=False, collate_fn=collate_fn, num_workers=0)

model = Model(model_type)
trainer = Trainer(model)
trainer.fit(train_loader=train_loader, out_dir='training1', epochs=epochs)
trainer.test(val_loader, 'training1')
