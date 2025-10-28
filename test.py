import argparse
from sand_ssl.io import CustomDataset, collate_wrapper
from sand_ssl.models import Extractor, Model
from torch.utils.data import DataLoader
from sand_ssl.training import Trainer
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', type=str, default='task1_train/training')
parser.add_argument('--spreadsheet_path', type=str, default='task1_train/sand_task_1.xlsx')
parser.add_argument('--metadata_path', type=str, default='task1_train/task1_metadata.csv')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--model_type', type=str, default="wavlm-large")
parser.add_argument('--batch_sz', type=int, default=2)
parser.add_argument('--output_path', type=str, default='sand_challenge')
args = parser.parse_args()

print(f'CUDA available: {torch.cuda.is_available()}')
training_data = CustomDataset(audio_dir=args.audio_dir, spreadsheet_path=args.spreadsheet_path, metadata_path=args.metadata_path, data_type='training', debug=args.debug)
validation_data = CustomDataset(audio_dir=args.audio_dir, spreadsheet_path=args.spreadsheet_path, metadata_path=args.metadata_path, data_type='validation', debug=args.debug)
#testing_data = CustomDataset(audio_dir=audio_dir2, spreadsheet_path=spreadsheet_path2, debug=False)

feature_extractor = Extractor(args.model_type, False)
collate_fn = collate_wrapper(feature_extractor)
train_loader = DataLoader(dataset=training_data, batch_size=args.batch_sz, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(dataset=validation_data,batch_size=args.batch_sz, shuffle=False, collate_fn=collate_fn, num_workers=0 )
#test_loader = DataLoader(dataset=testing_data, batch_size=batch_sz, shuffle=False, collate_fn=collate_fn, num_workers=0)

model = Model(model_type)
trainer = Trainer(model)
trainer.fit(train_loader=train_loader, out_dir=args.output_path, epochs=args.epochs)
trainer.test(val_loader, args.output_path)
