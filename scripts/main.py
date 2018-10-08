import glob
import numpy as np
import pandas as pd
from shapely.geometry import LineString,MultiLineString,Point,MultiPoint
from shapely.ops import linemerge
import pyproj
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import xgboost
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle
import os
import argparse

np.random.seed(10)

from param import *


#get an ensemble of 5 classifiers from scikit-learn i.e randome_forest, extra_tree,svc,KNeighbours 
#and xgboost classifier
#the parameters are tuned for this dataset, set class_weights to balanced as the start to end 
#goals have different distribution
def get_ensemble_of_classifiers(vote=True):
    clfs={}
    clf1=ExtraTreesClassifier(100,class_weight='balanced',n_jobs=-1)
    clfs['extra_tree']=clf1
    clf2=RandomForestClassifier(50,class_weight='balanced',n_jobs=-1)
    clfs['random_forest']=clf2
    clf3=KNeighborsClassifier(20,weights='distance',n_jobs=-1)
    clfs['knn']=clf3
    clf4=xgboost.XGBClassifier(n_estimators=100,subsample=.7)
    clfs['xgb']=clf4
    if vote:
        clf5=SVC(0.1)
        cvote=VotingClassifier(estimators=[('et', clf1), ('rf', clf2), ('kn', clf3),('xgb',clf4),('svc',clf5)], voting='hard')
        return {'cvote':cvote}
    else:
        clf5=SVC(0.1,class_weight='balanced',probability=True)
        clfs['svc']=clf5
        return clfs
  
# get the closest and farthest distance for a track to all the goals      
def closest_farthest(track):
    closest_to_track=[]
    farthest_to_track=[]
    for i in range(0,goal.shape[0]):
        point2=Point(goal[['lon','lat']].values[i])
        cd=[]
        for item in track:
            point1=Point(item)
            _,_,distance = geod.inv(point1.x, point1.y, point2.x, point2.y)
            cd.append(distance)
        closest_to_track.append(np.min(cd))
        farthest_to_track.append(np.max(cd))
    return closest_to_track,farthest_to_track

# get distance to a goal given a point on the track
def goal_dist(point1):
    d={}
    for i in range(0,goal.shape[0]):
        point2=Point(goal[['lon','lat']].values[i])
        angle1,angle2,distance = geod.inv(point1.x, point1.y, point2.x, point2.y)
        d[i]=distance
    return d.values()

# gets distance  features for training and testing 
# the feature vector includes  closest and nearest distances
# and distance to goal from the start or end points of track
def get_distances(df,goal,trim=None):
    start,end=Point(df[['lon','lat']].values[0]),Point(df[['lon','lat']].values[-1])
    duration=df.elapsedTime_sec.values[-1]
    _,_,total_distance_covered = geod.inv(start.x, start.y, end.x, end.y)
    distance_to_goal_from_start=goal_dist(start)
    distance_to_goal_from_end=goal_dist(end)
    closest,farthest=closest_farthest(df[['lon','lat']].values)
    return duration,total_distance_covered,distance_to_goal_from_start,distance_to_goal_from_end,closest,farthest

# similar to get_distance  function above  but additionally trims the start and end point randomly
def get_distances_multi(df,goal):
	# how much to trim from start
    trim_start=np.random.randint(TRIM_START,TRIM_END)
    idx_s=np.where(df.elapsedTime_sec>trim_start)[0][0]
    start=Point(df[['lon','lat']].values[idx_s])
    # how much to trim from end
    trim_end=np.random.randint(TRIM_START,TRIM_END)
    idx_e=np.where(df.elapsedTime_sec>df.elapsedTime_sec.values[-1]-trim_end)[0][0]
    end=Point(df[['lon','lat']].values[idx_e])
    _,_,total_distance_covered = geod.inv(start.x, start.y, end.x, end.y)
    distance_to_goal_from_start=goal_dist(start)
    distance_to_goal_from_end=goal_dist(end)
    duration=df.elapsedTime_sec.values[idx_e]
    closest,farthest=closest_farthest(df[['lon','lat']].values[idx_s:idx_e])
    return duration,total_distance_covered,distance_to_goal_from_start,distance_to_goal_from_end,closest,farthest
    
# get the train feature vector. The feature vector are aggressively augmented
# i.e for each feature vector 20 tracks with random trims are created from start and end
# also include other feature such as age, gender,duration,velocity and total distance covered
       
def get_train_feat(datafiles):
    print ('Multi trim featurees 20 samp in each')
    xfeat={}
    for f in tqdm(datafiles):
        for i in range(0,20):
            df = pd.read_csv(f)
            if i==0:
                duration,total_distance_covered,distance_to_goal_from_start,distance_to_goal_from_end,cd,fd=get_distances(df,goal,trim=None)
            else:
                duration,total_distance_covered,distance_to_goal_from_start,distance_to_goal_from_end,cd,fd=get_distances_multi(df,goal)
            feat=[duration,total_distance_covered]
            feat.extend(distance_to_goal_from_start)
            feat.extend(distance_to_goal_from_end)
            feat.extend(cd)
            feat.extend(fd)
            if df.tripID.values[0] not in xfeat.keys():
                xfeat[df.tripID.values[0]]=[feat]
            else:
                xfeat[df.tripID.values[0]].append(feat)
    train_info['gender']=pd.factorize(train_info['gender'])[0]
    train_info['age']=train_info['age'].fillna(train_info['age'].mean())
    features=[]
    labels_start=[]
    labels_end=[]
    for i,k in enumerate(train_info.tripID.values):
        for item in xfeat[k]:
            feat=train_info.loc[k][['age','gender']].values.tolist()
            duration=item[0]
            velocity=item[1]/duration
            feat.extend([duration,velocity])
            feat.extend(item)
            features.append(feat)
            labels_start.append(train_info.iloc[i]['startLocID'])
            labels_end.append(train_info.iloc[i]['destLocID'])         
    features=np.asarray(features).astype('float32')
    labels_start=np.asarray(labels_start).astype('int')
    labels_end=np.asarray(labels_end).astype('int')
    if SHUFFLE:
        idx=range(0,len(features))
        np.random.shuffle(idx)
        features,labels_start,labels_end=features[idx],labels_start[idx],labels_end[idx]
    return features,labels_start,labels_end

# get the test features...no augmentation as in the compition the features are already trimed    
def get_test_feat(datafiles):
    xfeat={}
    for f in tqdm(datafiles):
        df = pd.read_csv(f)
        duration,total_distance_covered,distance_to_goal_from_start,distance_to_goal_from_end,cd,fd=get_distances(df,goal,trim=None)
        feat=[duration,total_distance_covered]
        feat.extend(distance_to_goal_from_start)
        feat.extend(distance_to_goal_from_end)
        feat.extend(cd)
        feat.extend(fd)
        xfeat[df.tripID.values[0]]=feat
    test_info['gender']=pd.factorize(test_info['gender'])[0]
    test_info['age']=test_info['age'].fillna(test_info['age'].mean())
    features_test=[]
    for k in test_info.tripID.values:
        feat=test_info.loc[k][['age','gender']].values.tolist()
        duration=xfeat[k][0]
        velocity=xfeat[k][1]/duration
        feat.extend([duration,velocity])
        feat.extend(xfeat[k])
        features_test.append(feat)
    features_test=np.asarray(features_test).astype('float32')
    return features_test
    
# train the ensemble of classifiers    
def train_ens(features,slabels):
    sc=StandardScaler()
    sc.fit(features)
    clfs=get_ensemble_of_classifiers(vote=False)
    ft=sc.transform(features)
    for k in clfs:
        clfs[k].fit(ft,slabels)
        print ('train full data...done..with ',k)
    return sc,clfs

# predict from the ensemble and create submission
def submit_ens(clfs,features_test,ks,subname):
    y_pred=[]
    for key in clfs.keys():
        y_pred_i = clfs[key].predict_proba(features_test)
        y_pred.append(y_pred_i)
    y_pred = np.asarray(y_pred)
    y=np.mean(y_pred,axis=0)
    y_pred = np.argmax(y,axis=-1)
    preds = [list(ks[item]) for item in y_pred]
    np.savetxt(subname,preds, fmt='%d',delimiter=',')
    print ('done...')
    
# do cross val ensemble so we know what kind of score we will get
# note there is no weighting the tracks as in compition metric. simply get accuracy score and confusion matrix
def cross_val_ens(features,slabels,dirname):
    result={}
    clfs = get_ensemble_of_classifiers(vote=False)
    sc=StandardScaler()
    ft=sc.fit_transform(features)
    y_pred=[]
    for key in clfs.keys():
        y_pred_i = cross_val_predict(clfs[key], ft,slabels, cv=5,method='predict_proba')
        y_pred.append(y_pred_i)
        print ('cross val ...done...for ', key)
    y_pred=np.argmax(np.mean(y_pred,axis=0),axis=-1)
    score = accuracy_score(slabels,y_pred)
    result['start_acc']=score
    print("labels ens Accuracy: %0.4f " % score)
    conf_mat = confusion_matrix(slabels,y_pred)
    result['start_confusion']=conf_mat
    pickle.dump(result,open(''.join([dirname,'result.pkl']),'wb'))
    
# save the augmented feature for reproducability  also save ensemble
def save_aug_feat_model(features_train,labels_start,labels_end,features_test,clfs,dirname):
    features_dict={'feat_augs':[],'labels':[]}
    for i in range(0,len(features_train)):
        features_dict['feat_augs'].append(features_train[i])
        features_dict['labels'].append([labels_start[i],labels_end[i]])
    pickle.dump(features_dict,open(''.join([dirname,'features_train.pkl']),'wb'))
    pickle.dump(features_test,open(''.join([dirname,'features_test.pkl']),'wb'))
    if clfs is not None:
        pickle.dump(clfs,open(''.join([dirname,'clfs.pkl']),'wb'))
    print ('saved to...',dirname)

# There are 18 start-end pairs so we use this as training labels..ie 18 classes
# at test time compute the correspond start end label from predicted class
def get_combo_label(labels):
    ss=[tuple(item) for item in labels]
    s=set(ss)
    sk={}
    for i,k in enumerate(s):
        sk[k]=i
    slabel=[]
    for item in ss:
        slabel.append(sk[item])
    ks = {v: k for k, v in sk.items()}
    return slabel,ks
    
# do all augmented feature vector creation, training, converting start-end pair , saving and generating submission  
def train_ens_means(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    features_train,labels_start,labels_end=get_train_feat(train_datafiles)
    features_test=get_test_feat(test_datafiles)
    labels=np.vstack((labels_start,labels_end)).T
    slabels,ks=get_combo_label(labels)
    cross_val_ens(features_train,slabels,dirname)
    sc,clfs=train_ens(features_train,slabels)
    ft_test=sc.transform(features_test)
    save_aug_feat_model(features_train,labels_start,labels_end,features_test,clfs,dirname)
    submit_ens(clfs,ft_test,ks,''.join([dirname,'y_submission.txt']))

# generate submission from the feature vectors, labels and classifiers stored in a directoty
# can use same classifier or retrain
def test_from_dir(dirname,retrain=False):
    features_dict=pickle.load(open(''.join([dirname,'features_train.pkl']),'rb'))
    features_test=pickle.load(open(''.join([dirname,'features_test.pkl']),'rb'))
    features_train=np.asarray([item for item in features_dict['feat_augs']])
    labels_start=np.asarray([item[0] for item in features_dict['labels']])
    labels_end=np.asarray([item[1] for item in features_dict['labels']])
    labels=np.asarray([item for item in features_dict['labels']])
    slabels,ks=get_combo_label(labels)
    cross_val_ens(features_train,slabels,dirname)
    if retrain:
        print ('Retraining Ensembles')
        sc,clfs=train_ens(features_train,slabels)
        pickle.dump(clfs,open(''.join([dirname,'clfs.pkl']),'wb'))
    else:
        sc=StandardScaler()
        sc.fit(features_train)
        print ('Loading ensembles')
        clfs=pickle.load(open(''.join([dirname,'clfs.pkl']),'rb'))
    ft_test=sc.transform(features_test)
    submit_ens(clfs,ft_test,ks,''.join([dirname,'y_submission_retest.txt']))
   
# please specify the required command line options (refer to readme.txt for example)   
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train and test')
    parser.add_argument('--execute',default='train',help='Train or Test')
    parser.add_argument('--indir',required=True,help='Dir to read/write results')
    parser.add_argument('--retrain',default=False,help='Retrain from augmented features')
    args = parser.parse_args()
    if args.execute=='train':
        print ('Training and storing tmp results plus classifiers including y_submission.txt in {0}'.format(args.indir))
        if not os.path.exists(args.indir):
            os.mkdir(args.indir)
        else:
            print ('Results will be Replaced in {0} '.format(args.indir))
        train_ens_means(args.indir)
    if args.execute=='test':
        if not os.path.exists(args.indir):
            print ('{0} does not exists..'.format(args.indir))
        else:
            print ('Testing From {0} and write result in y_submission_retest.txt'.format(args.indir))
            test_from_dir(args.indir,args.retrain)
