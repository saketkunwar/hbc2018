import pandas as pd
import glob
import pyproj
import os

HOMEDIR='/media/saket/sk/hbc2018/'

if not os.path.exists(HOMEDIR):
	print ('HOMEDIR does not exists..')
	os.chdir('..')
	HOMEDIR = os.getcwd()
	print ('Setting HOMEDIR to {0} from current dir'.format(HOMEDIR))
	

train_info=pd.read_csv(''.join([HOMEDIR,'train_trip_info.csv']))
train_info.index=train_info.tripID
train_datafiles=glob.glob(''.join([HOMEDIR,'train/*.csv']))
goal=pd.read_csv(''.join([HOMEDIR,'goal_info.csv']))

test_datafiles=glob.glob(''.join([HOMEDIR,'test/*.csv']))
test_info=pd.read_csv(''.join([HOMEDIR,'test_trip_info.csv']))
test_info.index=test_info.tripID

geod = pyproj.Geod(ellps='WGS84')

TRIM_START=1
TRIM_END=60
SHUFFLE=False

