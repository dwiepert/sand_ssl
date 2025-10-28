
import pandas as pd

def _load_spreadsheet(spreadsheet_path, metadata_path, data_type, tasks, id_col):
    """
    """
    data = pd.read_excel(spreadsheet_path)

    if metadata_path:
        metadata = pd.read_csv(metadata_path)
        selected_spk = metadata[metadata['split'] == data_type]['spk_id'].to_list()
        data = data[data[id_col].isin(selected_spk)]
          
    tasks_expanded = [tasks] * len(data)



    data['task'] = tasks_expanded
    data = data.explode('task')
    data['uid'] = data[id_col] + '_' + data['task']
    data = data.set_index('uid')
    return data

