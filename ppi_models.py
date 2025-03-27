from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns

def ppi_prepoc(df,testproteins,classbalancing=None,test_size = 50,extra_feature=True):
    if extra_feature:
        df["aa_RelatPosition"] = df["aa_ProtPosition"] / df["Rlength"]
        df = df[[c for c in df if c not in ["domain","aa_ProtPosition","Rlength",'p_interface']] + ['p_interface']]
    else:
        df = df[[c for c in df if c not in ["domain","aa_ProtPosition","Rlength",'p_interface']] + ['p_interface']]

    # selecting Train and Test
    if testproteins == None:
        testproteins = random.sample(list(df["uniprot_id"].unique()),test_size)
    
    trainproteins = set(df["uniprot_id"].unique()) - set(testproteins) 
    
    df_train = df[df["uniprot_id"].isin(trainproteins)].sample(frac=1) # which protein_id's to use and sample to scramble rows
    df_test = df[df["uniprot_id"].isin(testproteins)].sample(frac=1)

    # Applying different sampling methods on the training data 

    if classbalancing == "under_sampling":
        new_x = df[df["p_interface"]==0].sample(7845)
        new_y = df[df["p_interface"]==1]
        df = pd.concat([new_x,new_y])
        df_train = df[df["uniprot_id"].isin(trainproteins)].sample(frac=1) # which protein_id's to use and sample to scramble rows
        df_test = df[df["uniprot_id"].isin(testproteins)].sample(frac=1)
        
        x_train = df_train[df_train.columns[:-1]].select_dtypes("number")
        y_train = df_train[df_train.columns[-1]]
    elif classbalancing == "over_sampling":
        # Concatenate the two groups
        X = df_train.select_dtypes("number").drop(columns=["p_interface"])  # Drop the target variable
        y = df_train["p_interface"]  # Target variable (p_interface)
        # Initialize SMOTE with the desired parameters
        smote = SMOTE(sampling_strategy='auto', random_state=1)
        x_train, y_train = smote.fit_resample(X, y)
    else:
        x_train = df_train[df_train.columns[:-1]].select_dtypes("number")
        y_train = df_train[df_train.columns[-1]]
            
    # making test data

    x_test = df_test[df_test.columns[:-1]].select_dtypes("number")
    y_test = df_test[df_test.columns[-1]]

    # Fitting the models
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(x_train, y_train)

    xgboost = XGBClassifier()
    xgboost.fit(x_train, y_train)
  
    # Displaying the results

    for model in [rf,xgboost]:
        #y_pred  = model.predict_proba(x_test)
        #curve  = roc_curve(y_test, y_pred[:, 1])
        #auc_  = auc(curve[0], curve[1])
        name = model.__str__().split('Class')[0]
        #plt.plot(curve[0], curve[1], label=f'{name} (area = {round(auc_,2)})')
        y_pred = model.predict(x_test)
        sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis")
        plt.suptitle(f"{name},Balancing:{classbalancing}",fontweight='bold')
        plt.title(f"Acc:{round(accuracy_score(y_test,y_pred),3)}, F1:{round(f1_score(y_test, y_pred),4)} extraF:{extra_feature}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.show()



    #plt.legend(bbox_to_anchor=(1,1))
    #plt.title(f'balancing:{classbalancing} test:{test_size}:proteins, feature extra{extra_feature}')
    #plt.show()
    
    feature_ranking = pd.DataFrame({"feature":x_train.columns,"importance":rf.feature_importances_})
    return feature_ranking.sort_values("importance",ascending=False,ignore_index=True)

def main():
    # read ppi as df
    # Removing unnamed Aminoacid
    df = pd.read_csv("ppi.csv") # reading in df
    df = df[df["sequence"]!= 'X'] # Removing the spooky amino-acid x
    df = df[df.columns[1:]]     # removing unnamed column

    # setting amount of testable proteins
    test_size = 50
    # running one instance of proteins to keep data comparable
    testproteins = random.sample(list(df["uniprot_id"].unique()),test_size)

    ppi_prepoc(df,testproteins,None,50,False)
    ppi_prepoc(df,testproteins,'under_sampling',False)
    ppi_prepoc(df,testproteins,'over_sampling',False)
    ppi_prepoc(df,testproteins,'under_sampling',True)

if __name__ == "__main__":
    main()
    