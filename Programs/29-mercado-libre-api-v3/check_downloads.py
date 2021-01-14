
#%%

import pandas as pd
import os

filenames = pd.Series(os.listdir('./parts'))
for count in ['arg', 'bra', 'col', 'chi', 'per', 'uru', 'ven', 'mex']:
	num = sum(filenames.str.contains(count,regex=False))
	args = filenames.str.findall(r'{}_part([0-9]+)'.format(count))
	max_files = args[args.str.len() != 0].apply(lambda x: x[0]).max()
	assert int(max_files) / 200 == num


# %%
