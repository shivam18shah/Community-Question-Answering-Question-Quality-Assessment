import pandas as pd
import os
import xml.etree.ElementTree as et
from config import args


def get_dataset(filepath: str, force_update: bool = False, debug: bool = False) -> object:
    """
    :param filepath: The filepath where the csv dataset is stored
    :param force_update: Forcefully get a new processed dataset
    :param debug: Print debug statements

    :return: A pandas dataframe containing the dataset
    """
    if (not os.path.exists(filepath)) or force_update:
        build_dataset(filepath, debug)

    if debug:
        print("Reading Dataset")

    return pd.read_csv(filepath)


def build_dataset(filepath: str, debug: bool = False) -> None:
    """
    :param filepath: The file path where the csv file should be saved
    :param debug: Print debug statements

    :return: Nothing
    """
    xtree = et.parse(os.path.join(args.dataset, "Posts.xml"))
    xroot = xtree.getroot()

    df_cols = ['Id', 'PostTypeId', 'AcceptedAnswerId', 'CreationDate', 'Score', 'ViewCount',
               'Body', 'Title', 'Tags', 'AnswerCount', 'CommentCount', 'FavoriteCount']

    rows = []
    if debug:
        print("Converting XML to CSV")

    for node in xroot:
        rows.append({
            'Id': node.attrib.get('Id'),
            'PostTypeId': node.attrib.get('PostTypeId'),
            'AcceptedAnswerId': node.attrib.get('AcceptedAnswerId'),
            'CreationDate': node.attrib.get('CreationDate'),
            'Score': node.attrib.get('Score'),
            'ViewCount': node.attrib.get('ViewCount'),
            'Body': node.attrib.get('Body'),
            'Title': node.attrib.get('Title'),
            'Tags': node.attrib.get('Tags'),
            'AnswerCount': node.attrib.get('AnswerCount'),
            'CommentCount': node.attrib.get('CommentCount'),
            'FavoriteCount': node.attrib.get('FavoriteCount')
        })

    out_df = pd.DataFrame(rows, columns=df_cols)

    out_df['AnswerCount'] = out_df['AnswerCount'].replace([None], ['0'])
    out_df['CommentCount'] = out_df['CommentCount'].replace([None], ['0'])
    out_df['FavoriteCount'] = out_df['FavoriteCount'].replace([None], ['0'])
    out_df['Score'] = out_df['Score'].replace([None], ['0'])
    out_df['ViewCount'] = out_df['ViewCount'].replace([None], ['0'])

    out_df = out_df.astype({
        'Id': 'int32',
        'PostTypeId': 'int32',
        'Score': 'int32',
        'ViewCount': 'int32',
        'CreationDate': 'datetime64[ns]',
        'AnswerCount': 'int32',
        'CommentCount': 'int32',
        'FavoriteCount': 'int32'
    })

    if debug:
        print("Pruning Dataset")

    # Dataset Pruning
    out_df = out_df[pd.notnull(out_df['Body'])]
    out_df = out_df[out_df['PostTypeId'] == 1]

    if debug:
        print("Saving Dataset")

    out_df.to_csv(filepath)