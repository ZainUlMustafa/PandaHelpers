import pandas as pd

def excel_to_json(excel_file_path, json_file_path):
    df = pd.read_excel(excel_file_path, engine='openpyxl')
    df = df.dropna()

    df.to_json(json_file_path, orient='records')

if __name__ == "__main__":
    excel_file_path = './data/dataset.xlsx'
    json_file_path = './out_data/dataset.json'

    excel_to_json(excel_file_path, json_file_path)
