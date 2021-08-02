import pandas as pd
from numpy import log, sqrt
from PreprocessingHelpers import integer_encoding, multiclass_list_encoding

vendors = pd.read_csv('vendors.csv')
orders = pd.read_csv('orders.csv')
train_customers = pd.read_csv('train_customers.csv')
train_locations = pd.read_csv('train_locations.csv')
test_customers = pd.read_csv('test_customers.csv')
test_locations = pd.read_csv('test_locations.csv')

#####################################
##         Process Orders          ##
#####################################

train_orders = orders[orders['customer_id'].isin(train_customers['akeed_customer_id'])]
test_orders = orders[orders['customer_id'].isin(test_customers['akeed_customer_id'])]



#####################################
##         Process Vendors         ##
#####################################

# Set id column to index
vendors.sort_values(by='id')
vendors.set_index('id', inplace=True)

# Fill primary_tags na with -1 & strip unnecessary characters
vendors['primary_tags'] = vendors['primary_tags'].fillna("{\"primary_tags\":\"-1\"}").apply(lambda x: int(str(x).split("\"")[3]))

# Fill vendor_tag na with -1 & convert to list-valued
vendors['vendor_tag'] = vendors['vendor_tag'].fillna(str(-1)).apply(lambda x: x.split(",")).apply(lambda x: [int(i) for i in x])

# Fix incorrect vendor_category_id
vendors.loc[28, 'vendor_category_id'] = 3.0

# Get unique vendor tags
vendor_tags = [int(tag) for tag in vendors['vendor_tag'].explode().unique()]
vendor_tags.sort()

# Map values to range(len(vendor_tags))
vendor_map = dict()
for i, tag in enumerate(vendor_tags):
    vendor_map[tag] = i
vendors['vendor_tag'] = vendors['vendor_tag'].apply(lambda tags: [vendor_map[tag] for tag in tags])

# Add num_orders, amt_sales, and avg_sale as new columns in vendor table
orders_vendor_grp = train_orders.groupby(by=['vendor_id'])
orders_per_vendor = orders_vendor_grp['akeed_order_id'].count().rename('num_orders')
grand_total_per_vendor = orders_vendor_grp['grand_total'].sum().rename('amt_sales')
vendors = vendors.merge(orders_per_vendor, how='left', left_on='id', right_index=True)
vendors = vendors.merge(grand_total_per_vendor, how='left', left_on='id', right_index=True)
vendors['avg_sale'] = vendors['amt_sales'] / vendors['num_orders']

# Log-transform to pad the skewness
vendors['num_orders_log3'] = vendors['num_orders'].apply(log).apply(log).apply(log)
vendors['amt_sales_log3'] = vendors['amt_sales'].apply(log).apply(log).apply(log)
vendors['avg_sale_log'] = vendors['avg_sale'].apply(log)

# Define location outliers
lat_lo, lat_hi = -25, 25
long_lo, long_hi = -5, 5
v_outliers = (vendors['latitude'] < lat_lo) | (vendors['latitude'] > lat_hi) | (vendors['longitude'] < long_lo) | (vendors['longitude'] > long_hi)

# Project vendor outliers
for i in vendors[v_outliers].index:
        lat = vendors.loc[i, 'latitude']
        long = vendors.loc[i, 'longitude']
        mag = sqrt(lat**2 + long**2)
        vendors.loc[i, 'latitude'] = lat / mag * lat_hi
        vendors.loc[i, 'longitude'] = long / mag * long_hi

# Choose which columns to use
keep_continuous = ['latitude', 'longitude', 'delivery_charge', 'serving_distance', 'prepration_time', 'vendor_rating', 'num_orders_log3', 'amt_sales_log3', 'avg_sale_log']
keep_categorical = ['vendor_category_id', 'status', 'rank', 'primary_tags', 'vendor_tag']
keep_columns = keep_continuous + keep_categorical
vendors = vendors[keep_columns]

# Encode categorical columns
vendors, _ = integer_encoding(df=vendors, cols=['vendor_category_id', 'delivery_charge', 'status', 'rank', 'primary_tags'], drop_old=True, monotone_mapping=True)
vendors = multiclass_list_encoding(df=vendors, cols=['primary_tags', 'vendor_tag'], drop_old=True)



#####################################
##         Process Customers       ##
#####################################













