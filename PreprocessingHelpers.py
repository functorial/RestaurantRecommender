import pandas as pd

def get_sequences(df:pd.DataFrame, target:str, group_by:list, sort_by:str=None, sort:bool=False, min_seq_len=1) -> pd.Series:
    """Groups a DataFrame by features and aggregates target feature into a Series of lists."""
    clone = df.copy()
    if sort:
        clone.sort_values(by=sort_by, inplace=True)
    group = clone.groupby(by=group_by)
    sequences = group[target].apply(list)
    sequences = sequences[sequences.apply(lambda x: len(x)) >= min_seq_len]    # Filter out locations with 0 orders
    return sequences

def integer_encoding(df:pd.DataFrame, cols:list, drop_old=False, sort_unique:bool=False, ascending=True):
    """Returns updated DataFrame and inverse mapping dictionary."""
    for col in cols:
        unique_values = df[col].unique()
        if sort_unique:
            unique_values.sort_values(inplace=True, ascending=ascending)
        num_unique = unique_values.size
        id_map = dict()
        inv_map = dict()
        for i in range(num_unique):
            id_map[unique_values.iloc[i]] = i
            inv_map[i] = unique_values.iloc[i]
        col_reidx = col + "_reidx"
        df[col_reidx] = df[col].map(id_map)
        if drop_old:
            df.drop(labels=cols)
        return df, inv_map
