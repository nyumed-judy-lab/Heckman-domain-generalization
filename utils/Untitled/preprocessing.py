import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def DatasetImporter_tabular(train_val_dat, args, num_cols, cat_cols):
    train_sites = sorted(args.train_domains)
    train_dat, val_dat = train_test_split(train_val_dat, stratify=train_val_dat['SITE'] + train_val_dat['CVD'].astype(str), test_size=args.test_size, random_state=args.seed)

    train_idx = [s in train_sites for s in train_dat['SITE'].tolist()]
    valid_idx = [s in train_sites for s in val_dat['SITE'].tolist()]

    tr_s = train_dat[train_idx]['SITE']
    tr_x = train_dat[train_idx][num_cols].values.astype(float)
    tr_y = train_dat[train_idx]['CVD'].values.astype(float)
    

    val_s = val_dat[valid_idx]['SITE']
    val_x = val_dat[valid_idx][num_cols].values.astype(float)
    val_y = val_dat[valid_idx]['CVD'].values.astype(float)

    num_imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    num_imputer = num_imputer.fit(tr_x)
    scaler = scaler.fit(tr_x)
    
    tr_x = num_imputer.fit_transform(tr_x)
    tr_x = scaler.fit_transform(tr_x)
    val_x = num_imputer.transform(val_x)
    val_x = scaler.transform(val_x)

    tr_x = np.column_stack([tr_x, train_dat[train_idx][cat_cols].values.astype(float)])
    val_x = np.column_stack([val_x, val_dat[valid_idx][cat_cols].values.astype(float)])
    
    return tr_x, tr_s, tr_y, val_x, val_s, val_y, num_imputer, scaler