import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score

def get_univariate_table(estimator_list, X_train, name_list, y_train, cv = None):
    table=[]
    annot_table=[]
    for estimator in estimator_list:
        list = []
        annot_list = []
        for feature in X_train.columns:
            Accs = cross_val_score(estimator,
                                   X_train[[feature]],
                                   y_train, 
                                   cv = cv)
            mean_Acc = np.mean(Accs)
            std_Acc = np.std(Accs)
            list.append(mean_Acc)
            annot_list.append(f'{mean_Acc:.3f} ± {std_Acc:.3f}')    
        table.append(list)
        annot_table.append(annot_list)
    table = pd.DataFrame(table, index = name_list, columns = X_train.columns)
    annot_table = pd.DataFrame(annot_table)
    return table, annot_table

def plot_univariate_auc(table, annot_table):
    ax = sns.heatmap(data = table.transpose(),
                     annot = annot_table.transpose(),
                     fmt = '', 
                     cmap= 'coolwarm',
                     center=0.55)
    ax.set_ylabel('Features\n')
    ax.set_title('Univariate Accuracy in Models\n')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    fig = ax.get_figure()
    return fig
