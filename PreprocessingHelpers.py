import pandas as pd
import torch

def get_sequences(df:pd.DataFrame, target:str, group_by:list, sort_by:str=None, sort:bool=False, min_seq_len=1) -> pd.Series:
    """Groups a DataFrame by features and aggregates target feature into a Series of lists."""
    clone = df.copy()
    if sort:
        clone.sort_values(by=sort_by, inplace=True)
    group = clone.groupby(by=group_by)
    sequences = group[target].apply(list)
    sequences = sequences[sequences.apply(lambda x: len(x)) >= min_seq_len]    # Filter out length 0 sequences
    return sequences

def integer_encoding(df:pd.DataFrame, cols:list, min_int=0, drop_old=False, monotone_mapping:bool=False):
    """Returns updated DataFrame and inverse mapping dictionary."""
    clone = df.copy()
    id_maps = dict()
    inv_maps = dict()
    for col in cols:
        # If list-valued
        if type(clone.iloc[0][col]) == list:
            # Get unique values and sort
            unique_values = clone[col].explode().unique()
            num_unique = unique_values.size
            if monotone_mapping:
                unique_values.sort()
            # Generate dictionary maps
            id_map = dict()
            inv_map = dict()
            for i in range(num_unique):
                id_map[unique_values[i]] = i + min_int
                inv_map[i + min_int] = unique_values[i]
            id_maps[col] = id_map
            inv_maps = inv_map
            # Encoding
            if drop_old:
                clone[col] = clone[col].apply(lambda x: [id_map[i] for i in x])
            else:
                col_reidx = col + "_reidx"
                clone[col_reidx] = clone[col].apply(lambda x: [id_map[i] for i in x])
        else:
            # Get unique values and sort
            unique_values = clone[col].unique()
            num_unique = unique_values.size
            if monotone_mapping:
                unique_values.sort()
            # Generate dictionary maps
            id_map = dict()
            inv_map = dict()
            for i in range(num_unique):
                id_map[unique_values[i]] = i + min_int
                inv_map[i + min_int] = unique_values[i]
            id_maps[col] = id_map
            inv_maps[col] = inv_map
            # Encoding
            if drop_old:
                clone[col] = clone[col].map(id_map)
            else:
                col_reidx = col + "_reidx"
                clone[col_reidx] = clone[col].map(id_map)
    return clone, id_maps, inv_maps

def multiclass_list_encoding(df:pd.DataFrame, cols:list, drop_old=False):
    clone = df.copy()

    # For index conjugation to make querying easy
    index_map = dict()
    inv_map = dict()
    for i, idx in enumerate(clone.index):
        index_map[idx] = i
        inv_map[i] = idx
        
    clone.index = clone.index.map(index_map)

    for col in cols:
        # If list-valued
        if type(clone.iloc[0][col]) == list:
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

    clone.index = clone.index.map(inv_map)
    return clone


def pool_encodings_from_sequences(sequences:pd.Series, pool_from: pd.DataFrame):
    """Inputs a Pandas Series `sequences` valued in lists of indices from `pool_from`.
    Outputs a Pandas DataFrame with columns from `pool_from` and indices from `sequences`
    with values given as a mean over `pool_from` rows supplied from `sequences`."""
    encoded = pd.DataFrame(index=sequences.index, columns=pool_from.columns, dtype='float64')
    seq_df = sequences.to_frame()
    col = seq_df.columns[0]
    def f(row):
        seq = row[col]
        encoded.loc[row.name] = pool_from[pool_from.index.isin(seq)].mean(axis=0)
        return None
    seq_df.apply(f, axis=1)
    return encoded


def get_inputs_from_sequences(sequences:pd.Series, customers:pd.DataFrame, vendors:pd.DataFrame):
    out = torch.zeros((1, 2 * len(customers.columns)))  # For shape
    seq_df = sequences.to_frame()
    col = seq_df.columns[0]
    def f(row):
        seq = row[col]
        c_tensor = torch.tensor(customers.loc[row.name])
        for vendor in seq:
            v_tensor = torch.tensor(vendors.iloc[vendor])
            pair = torch.cat((c_tensor, v_tensor)).view(1, -1)
            out = torch.cat((out, pair), axis=0)
        return None
    seq_df.apply(f, axis=1)
    return out[1:]


def pandas_sequences_to_tensor(sequences:pd.Series, window:int=5):

    def left_pad_list(L):
        nonlocal window
        num_zeros = window - len(L)
        return ([0] * num_zeros) + L

    def get_windows(L):
        nonlocal window
        out = list()
        for i in range(1, len(L)+1):
            if i <= window:
                out.append(left_pad_list(L[:i]))
            else:
                out.append(L[i-window:i])
        return out

    padded_sequences = torch.stack(sequences.apply(get_windows).explode().apply(torch.tensor).tolist(), axis=0)
    return padded_sequences


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.customer = sequences[:, :-1]
        self.vendor = sequences[:, -1:].view(-1)

    def __len__(self):
        return len(self.vendor)

    def __getitem__(self, idx):
        return self.customer[idx,:], self.vendor[idx]