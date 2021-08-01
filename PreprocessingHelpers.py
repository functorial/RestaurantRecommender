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


def integer_encoding(df:pd.DataFrame, cols:list, drop_old=False, monotone_mapping:bool=False, ascending=True):
    """Returns updated DataFrame and inverse mapping dictionary."""
    clone = df.copy()
    inv_maps = dict()
    for col in cols:
        unique_values = clone[col].unique()
        if monotone_mapping:
            unique_values.sort()
        num_unique = unique_values.size
        id_map = dict()
        inv_map = dict()
        for i in range(num_unique):
            id_map[unique_values[i]] = i
            inv_map[i] = unique_values[i]
        inv_maps[col] = inv_map
        if drop_old:
            clone[col] = clone[col].map(id_map)
        else:
            col_reidx = col + "_reidx"
            clone[col_reidx] = clone[col].map(id_map)
    return clone, inv_maps

def multiclass_list_encoding(df:pd.DataFrame, cols:list, drop_old=False):
    clone = df.copy()
    for col in cols:

        # If list-valued
        if type(clone.loc[0, col]) == list:
            categories = clone[col].explode().unique().tolist()
            categories.sort()
            # Init one-hot columns
            for cat in categories:
                cat_col = col + "_is_" + str(cat)
                clone[cat_col] = 0
            # Define encoding function to be vectorized
            def f(row):
                row_cats = row[col]     # type(row_cats) == list
                for row_cat in row_cats:
                    row_cat_col = col + "_is_" + str(row_cat)
                    idx = row.name
                    clone.loc[idx, row_cat_col] = 1
            clone.apply(f, axis=1)

        # If not list-valued
        else:
            categories = clone[col].unique().tolist()
            categories.sort()
            # Init one-hot columns
            for cat in categories:
                cat_col = col + "_is_" + str(cat)
                clone[cat_col] = 0
            # Define encoding function to be vectorized
            def g(row):
                row_cat = row[col]
                row_cat_col = col + "_is_" + str(row_cat)
                idx = row.name
                clone.loc[idx, row_cat_col] = 1
            clone.apply(g, axis=1)

    if drop_old:
        clone.drop(labels=cols, axis=1, inplace=True)
    return clone

        