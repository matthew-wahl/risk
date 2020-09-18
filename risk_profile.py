import pandas as pd
import numpy as np
import math

# funds in dollars
# funds = 15000

# df = pd.read_csv(''C:/Users/.../td_webroker_watchlist.CSV', header=5)
df1 = df[['Name', 'Market', 'Description', 'Last Price', '52 Week Low', '52 Week High']]

security_list = df1.values.tolist()
# commission
funds = funds - len(security_list) * 20

# append percent change
for security in security_list:
    lo_diff = (security[4] - security[3]) / security[3]
    security.append(round(lo_diff, 4))
    hi_diff = (security[5] - security[3]) / security[3]
    security.append(round(hi_diff, 4))

# Solve so that each security gains/loses the same if they all hit their respective 52-week edge case
# 1. put into system of equations
#       -matrix A is L.S, matrix b is R.S
#       -the first row is alpha + beta + gamma + ... = F
#       -the following rows solve for each:
#                                       A0*alpha = B0*beta
#                                       A0*alpha = G0*gamma
#                                                .
#                                                .
#                                                .
#                                       A0*alpha = N0*nu
#
#       - represented in matrix form:   A0*alpha - B0*beta = 0
#                                       A0*alpha - G0*gamma = 0
#                                                .
#                                                .
#                                                .
#                                       A0*alpha - N0*nu = 0
# 2. solve system

solutions = []
# start with 52-lo
greek = 6

# run for both
for _ in range(0, 2):
    variables = []
    for security in security_list:
        variables.append(security[greek])

    # create A and b
    b = [0] * len(variables)
    b[0] = funds
    b = np.array(b)

    A = [[0 for x in variables] for y in variables]
    column = 0
    for row in A:
        row[column] = -variables[column]
        row[0] = variables[0]
        column += 1
    A[0] = [1 for x in A[0]]
    A = np.array(A)

    x = np.linalg.solve(A, b)

    x_list = x.tolist()
    solutions.append(x_list)

    greek += 1

equal_loss_list = [solutions[0]]
equal_gain_list = [solutions[1]]
last_price = df1['Last Price'].values.tolist()
equal_loss_list.append(last_price)
equal_gain_list.append(last_price)

number_of_shares = []
for x in equal_loss_list[0]:
    number_of_shares.append(math.floor(x / equal_loss_list[1][equal_loss_list[0].index(x)]))
equal_loss_list.append(number_of_shares)

number_of_shares = []
for x in equal_gain_list[0]:
    number_of_shares.append(math.floor(x / equal_gain_list[1][equal_gain_list[0].index(x)]))
equal_gain_list.append(number_of_shares)

max_loss = []
for x in security_list:
    percent = security_list[security_list.index(x)][6]
    max_loss.append(percent)
equal_loss_list.append(max_loss)

max_gain = []
for x in security_list:
    percent = security_list[security_list.index(x)][7]
    max_gain.append(percent)
equal_loss_list.append(max_gain)

max_loss = []
for x in security_list:
    percent = security_list[security_list.index(x)][6]
    max_loss.append(percent)
equal_gain_list.append(max_loss)

max_gain = []
for x in security_list:
    percent = security_list[security_list.index(x)][7]
    max_gain.append(percent)
equal_gain_list.append(max_gain)

# max gain and max loss
# equal loss
min_value = []
difference = []
for x in equal_loss_list[0]:
    # price * number_of_shares * percent loss
    price = equal_loss_list[1][equal_loss_list[0].index(x)]
    number_of_shares = equal_loss_list[2][equal_loss_list[0].index(x)]
    old_value = price * number_of_shares
    percent_loss = equal_loss_list[3][equal_loss_list[0].index(x)]
    new_price = (price * percent_loss) + price
    new_value = new_price * number_of_shares
    min_value.append(new_value)
    value_change = new_value - old_value
    difference.append(value_change)
equal_loss_list.append(min_value)
equal_loss_list.append(difference)

max_value = []
difference = []
for x in equal_loss_list[0]:
    # price * number_of_shares * percent loss
    price = equal_loss_list[1][equal_loss_list[0].index(x)]
    number_of_shares = equal_loss_list[2][equal_loss_list[0].index(x)]
    old_value = price * number_of_shares
    percent_gain = equal_loss_list[4][equal_loss_list[0].index(x)]
    new_price = (price * percent_gain) + price
    new_value = new_price * number_of_shares
    max_value.append(new_value)
    value_change = new_value - old_value
    difference.append(value_change)
equal_loss_list.append(max_value)
equal_loss_list.append(difference)

# equal gain
min_value = []
difference = []
for x in equal_gain_list[0]:
    # price * number_of_shares * percent loss
    price = equal_gain_list[1][equal_gain_list[0].index(x)]
    number_of_shares = equal_gain_list[2][equal_gain_list[0].index(x)]
    old_value = price * number_of_shares
    percent_loss = equal_gain_list[3][equal_gain_list[0].index(x)]
    new_price = (price * percent_loss) + price
    new_value = new_price * number_of_shares
    min_value.append(new_value)
    value_change = new_value - old_value
    difference.append(value_change)
equal_gain_list.append(min_value)
equal_gain_list.append(difference)

max_value = []
difference = []
for x in equal_gain_list[0]:
    # price * number_of_shares * percent loss
    price = equal_gain_list[1][equal_gain_list[0].index(x)]
    number_of_shares = equal_gain_list[2][equal_gain_list[0].index(x)]
    old_value = price * number_of_shares
    percent_gain = equal_gain_list[4][equal_gain_list[0].index(x)]
    new_price = (price * percent_gain) + price
    new_value = new_price * number_of_shares
    max_value.append(new_value)
    value_change = new_value - old_value
    difference.append(value_change)
equal_gain_list.append(max_value)
equal_gain_list.append(difference)

old_value = []
for x in equal_gain_list[0]:
    # price * number_of_shares * percent loss
    price = equal_gain_list[1][equal_gain_list[0].index(x)]
    number_of_shares = equal_gain_list[2][equal_gain_list[0].index(x)]
    old_value.append(price * number_of_shares)
equal_gain_list.append(old_value)

old_value = []
for x in equal_loss_list[0]:
    # price * number_of_shares * percent loss
    price = equal_loss_list[1][equal_loss_list[0].index(x)]
    number_of_shares = equal_loss_list[2][equal_loss_list[0].index(x)]
    old_value.append(price * number_of_shares)
equal_loss_list.append(old_value)

name = df['Name'].values.tolist()
market = df['Market'].values.tolist()
description = df['Description'].values.tolist()

df2 = pd.DataFrame(data=equal_loss_list)
df3 = pd.DataFrame(data=equal_gain_list)
df_equal_loss = df2.transpose()
df_equal_gain = df3.transpose()

cols = ['Fund allocation', 'Price',
        'Number of Shares', 'Max Loss Percent', 'Max Gain Percent', 'Min Value',
        'Max Loss Difference', 'Max Value', 'Max Gain Difference', 'Original Value']

df_equal_loss.columns = cols
df_equal_gain.columns = cols

df_equal_loss['Name'] = name
df_equal_loss['Market'] = market
df_equal_loss['Description'] = description

df_equal_gain['Name'] = name
df_equal_gain['Market'] = market
df_equal_gain['Description'] = description

cols = ['Name', 'Market', 'Description', 'Fund allocation', 'Price',
        'Number of Shares', 'Original Value', 'Max Loss Percent', 'Min Value',
        'Max Loss Difference', 'Max Gain Percent', 'Max Value', 'Max Gain Difference']

df_equal_gain = df_equal_gain[cols]
df_equal_loss = df_equal_loss[cols]

pd.options.display.max_columns = None
pd.options.display.max_rows = None

equal_gain_max_loss = df_equal_gain['Min Value'].sum()
equal_gain_max_gain = df_equal_gain['Max Value'].sum()
equal_gain_r = equal_gain_max_gain/equal_gain_max_loss

equal_loss_max_loss = df_equal_loss['Min Value'].sum()
equal_loss_max_gain = df_equal_loss['Max Value'].sum()
equal_loss_r = equal_loss_max_gain/equal_loss_max_loss

print(f'equal gain: {equal_gain_r}')
print(f'equal loss: {equal_loss_r}')

df1.to_csv('C:/Users/..../original_data.CSV', index=False)
df_equal_gain.to_csv('C:/Users/.../equal_gain.CSV', index=False)
df_equal_loss.to_csv('C:/Users/.../equal_loss.CSV', index=False)