import os
path = '/Users/pek2012/Desktop/STORK/Images/test'
filenames = os.listdir(path)

for filename in filenames:
    os.rename(os.path.join(path, filename), os.path.join(path, filename.replace('_', '')))