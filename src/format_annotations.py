import os
import pandas as pd
from pathlib import Path
import argparse

def check_for_missing_data(df:pd.DataFrame):
    missing_rows = []
    for row in df.index:
        if len(df.result.loc[row]) == 0:
            missing_rows.append(row)

    return missing_rows

def format_annot(df:pd.DataFrame):

    ## Remove unused columns
    df = df[['annotations', 'file_upload']]

    ## Format the file names
    # Get the index for the "-" in the file name
    dash_ind = df.file_upload[0].index('-')
    # Filter the file_upload string to only contain the file name
    df = df.assign(file_name = df.file_upload.str[dash_ind+1:])

    # Get the annotations dictionary only
    result_df = df.annotations.apply(lambda x: pd.Series(x[0]))

    # Check for missing rows
    missing_rows = check_for_missing_data(result_df)
    # Drop rows with missing values
    result_df = result_df.drop(missing_rows)

    result_df = result_df.result.apply(lambda x: pd.Series(x[0]))
    # Get the value dictionary only
    value_df = result_df.value.apply(pd.Series)

    # Merge everything into a final dataframe
    final_df = df
    final_df = final_df.assign(
    original_width = result_df.original_width,
    original_height = result_df.original_height,
    x = value_df.x,
    y = value_df.y)

    # Drop unused columns
    final_df = final_df.drop(['annotations', 'file_upload'], axis = 1)

    return final_df

if __name__ == '__main__':

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot-path', type=str)
    opt = parser.parse_args()

    cwd = Path.cwd()
    main_folder = cwd.parent
    # Raw annotations files path
    curr_annot_path = main_folder / opt.annot_path
    # List the files in the folder
    lst_annot_files = os.listdir(curr_annot_path)
    # Get only the FULL annotation files
    lst_annot_files = [file for file in lst_annot_files if file.endswith('_FULL.json')]
    # Output path
    final_annot_path = main_folder / 'curr_annotations'

    # Concatenate the pandas data
    dfs = []
    for file in lst_annot_files:
        df_import = pd.read_json(curr_annot_path / file)
        dfs.append(df_import)

    df = pd.concat(dfs, ignore_index=True)

    # Perform the formatting
    df = format_annot(df)

    # Drop any duplicate file labels
    df = df.drop_duplicates(subset='file_name', keep='first')

    print(f'Annotations shape: {df.shape}')

    # Save the formatted dataframe to json
    df.to_json(final_annot_path / 'annotations.json')
