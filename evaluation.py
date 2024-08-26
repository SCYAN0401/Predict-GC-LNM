import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import shap

from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix,     
    accuracy_score, 
    precision_score, 
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    brier_score_loss
)

from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

### random_list
def make_random_list(n_samples = 1000, seed = None):
    rng = np.random.default_rng(seed)
    random_list = list(rng.choice(np.arange(n_samples*10), size = n_samples, replace = False))
    return random_list

### metrics HM
def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity_score = tp / (tp + fn)
    return sensitivity_score

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity_score = tn / (tn + fp)
    return specificity_score

def evaluation_report(X_list, y, estimator_list, name_list, n_samples = 1000, random_list = None):
    table = []
    annot_table = []
    score_list = [
        accuracy_score, 
        precision_score, 
        f1_score,
        sensitivity_score, 
        specificity_score, 
        # roc_auc_score, 
        # average_precision_score,  
        # brier_score_loss
    ]
    sname_list = [
        'Accuracy',
        'Precision',
        'F1 score',
        'Sensitivity',
        'Specificity',
        # 'AUC score',
        # 'AP score',
        # 'Brier score',
    ]
    
    y_true = np.array(y).ravel()
    
    for j, (estimator, X) in enumerate(zip(estimator_list, X_list)):
        y_pred = estimator.predict(X)
        y_prob = estimator.predict_proba(X)[:, 1]
        
        results_list = []
        
        list = []
        annot_list = []
   
        for metric in score_list:
            for i in range(n_samples):
                indices = resample(np.arange(len(y_true)), random_state = random_list[i])
                
                if any(metric.__name__ == a for a in ['roc_auc_score', 'average_precision_score', 'brier_score_loss']):                    
                    result = metric(y_true[indices], y_prob[indices])
                else:
                    result = metric(y_true[indices], y_pred[indices])
                
                results_list.append(result)

            mean_metric = np.mean(results_list, axis=0)
            ci_metric = np.percentile(results_list, [2.5, 97.5], axis=0)
            
            annot_metric = (f'{mean_metric:.3f}\n{ci_metric[0]:.3f}-{ci_metric[1]:.3f}')
            list.append(mean_metric)
            annot_list.append(annot_metric)
            
        table.append(list)
        annot_table.append(annot_list)
        
    table = pd.DataFrame(table, index = name_list, columns = sname_list)
    annot_table = pd.DataFrame(annot_table)
    return table, annot_table


def plot_evaluation_report(table, annot_table):
    ax = sns.heatmap(data = table.transpose(),
                     annot = annot_table.transpose(),
                     fmt = '', 
                     cmap = 'coolwarm',
                     center = 0.65)
    ax.set_ylabel('Metrics with 95% CI\n')
    ax.set_title('Model Performance\n')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    fig = ax.get_figure()
    return fig

### ROC curves

def plot_roc_curve_together(X_list, y, estimator_list, name_list, n_samples = 1000, random_list = None):
    y_true = np.array(y).ravel()
    
    plots_mean = []
    legend_names = []

    for j, (estimator, X, name) in enumerate(zip(estimator_list, X_list, name_list)):

        y_prob = estimator.predict_proba(X)[:, 1]
        fprs = []
        tprs = []
        roc_aucs = []

        for i in range(n_samples):
            
            indices = resample(np.arange(len(y_true)), random_state = random_list[i])
            fpr, tpr, _ = roc_curve(y_true[indices], y_prob[indices])
            
            fprs.append(fpr)
            tprs.append(tpr)

            roc_auc = roc_auc_score(y_true[indices], y_prob[indices])
            roc_aucs.append(roc_auc)

        fprs = pd.DataFrame(fprs)
        mean_fpr = np.mean(fprs, axis=0)

        tprs = pd.DataFrame(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        
        mean_roc_auc = np.mean(roc_aucs, axis=0)        
        ci_roc_auc = np.percentile(roc_aucs, [2.5, 97.5], axis=0)

        plot_mean = RocCurveDisplay(
            fpr = mean_fpr, 
            tpr = mean_tpr, 
            roc_auc = None,
            estimator_name = None
        )
        plots_mean.append(plot_mean)
        legend_names.append(f"{name} (AUC = {mean_roc_auc:.3f}, {ci_roc_auc[0]:.3f}-{ci_roc_auc[1]:.3f})")
        
    return plots_mean, legend_names

### PR curves

def plot_pr_curve_together(X_list, y, estimator_list, name_list, n_samples = 1000, random_list = None):
    y_true = np.array(y).ravel()
    
    plots_mean = []
    legend_names = []

    for j, (estimator, X, name) in enumerate(zip(estimator_list, X_list, name_list)):

        y_prob = estimator.predict_proba(X)[:, 1]
        precs = []
        recalls = []
        ap_aucs = []

        for i in range(n_samples):
            
            indices = resample(np.arange(len(y_true)), random_state = random_list[i])
            prec, recall, _ = precision_recall_curve(y_true[indices], y_prob[indices])
            
            precs.append(prec)
            recalls.append(recall)

            ap_auc = average_precision_score(y_true[indices], y_prob[indices])
            ap_aucs.append(ap_auc)

        precs = pd.DataFrame(precs)
        mean_prec = np.mean(precs, axis=0)

        recalls = pd.DataFrame(recalls)
        mean_recall = np.mean(recalls, axis=0)
        
        mean_ap_auc = np.mean(ap_aucs, axis=0)        
        ci_ap_auc = np.percentile(ap_aucs, [2.5, 97.5], axis=0)

        plot_mean = PrecisionRecallDisplay(
            mean_prec, 
            mean_recall, 
        )
        plots_mean.append(plot_mean)
        legend_names.append(f"{name} (AP = {mean_ap_auc:.3f}, {ci_ap_auc[0]:.3f}-{ci_ap_auc[1]:.3f})")
        
    return plots_mean, legend_names

### DCA

def plot_dca_together(X_list, y, estimator_list, name_list, n_samples = 1000, random_list = None, ax = None):
    y_true = np.array(y).ravel()
    thresh_group = np.arange(0,1,0.01)


    ### treat all
    
    net_benefit_all = np.array([])
    tn_, fp_, fn_, tp_ = confusion_matrix(y_true, y_true).ravel()
    total = tp_ + tn_

    for thresh in thresh_group:
        net_benefit_ = (tp_ / total) - (tn_ / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit_)
            
    ax.plot(thresh_group, 
            net_benefit_all, 
            color = 'black',
            label = 'Treat all')

    ### treat none

    ax.axhline(y = 0.0,  
                color = 'black', 
                linestyle = 'dotted', 
                label = 'Treat none')


    for j, (X, estimator, name) in enumerate(zip(X_list, estimator_list, name_list)):
        pred_proba = estimator.predict_proba(X)[:, 1]
        net_benefit_model_bootstrap = []

        for i in range(n_samples):
            indices = resample(np.arange(len(y_true)), random_state = random_list[i])
            net_benefit_model = np.array([])

            for thresh in thresh_group:
                y_pred_label = pred_proba[indices] > thresh
                tn, fp, fn, tp = confusion_matrix(y_true[indices], y_pred_label).ravel()
                n = len(y_true[indices])
                net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
                net_benefit_model = np.append(net_benefit_model, net_benefit)
            net_benefit_model_bootstrap.append(net_benefit_model)

        mean_net_benefit_model = np.mean(net_benefit_model_bootstrap, axis = 0)

        ax.plot(thresh_group,
                 mean_net_benefit_model,
                 label = name)
    
    ### aes

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Mean DCA")

    ax.legend(loc = 'upper right')
    return ax

### calibration curves

def plot_calibration_curve_together(X_list, y, estimator_list, name_list, dots_no = 101, n_samples = 1000, random_list = None):
    y_true = np.array(y).ravel()
    plots_mean = []
    plots_ci_lower = []
    plots_ci_upper = []
    legend_names = []
    
    for j, (estimator, X, name) in enumerate(zip(estimator_list, X_list, name_list)):

        y_prob = estimator.predict_proba(X)[:, 1]
        prob_trues = []
        prob_preds = []
        y_probs = []

        for i in range(n_samples):
            indices = resample(np.arange(len(y_true)), random_state = random_list[i])
            prob_true, prob_pred = calibration_curve(y_true[indices], 
                                                     y_prob[indices],
                                                     n_bins = dots_no, 
                                                     strategy='quantile')
            prob_trues.append(prob_true)
            prob_preds.append(prob_pred)
            y_probs.append(y_prob[indices])

        prob_trues = pd.DataFrame(prob_trues)
        mean_prob_true = np.mean(prob_trues, axis=0)
        
        prob_preds = pd.DataFrame(prob_preds)
        mean_prob_pred = np.mean(prob_preds, axis=0)

        y_probs = pd.DataFrame(y_probs)
        mean_y_prob = np.mean(y_probs, axis=0)

        plot_mean = CalibrationDisplay(mean_prob_true, mean_prob_pred, mean_y_prob)
        plots_mean.append(plot_mean)
        legend_names.append(f'{name}')

    return plots_mean, legend_names

### shap
    
def plot_beeswarm_per_features(explanation, name):
    mask = [name in n for n in explanation.feature_names]
    explanation_ = shap.Explanation(explanation.values[:, mask],
                                    feature_names=list(np.array(explanation.feature_names)[mask]),
                                    data=explanation.data[:, mask],
                                    base_values=explanation.base_values,
                                    display_data=explanation.display_data,
                                    instance_names=explanation.instance_names,
                                    output_names=explanation.output_names,
                                    output_indexes=explanation.output_indexes,
                                    lower_bounds=explanation.lower_bounds,
                                    upper_bounds=explanation.upper_bounds,
                                    main_effects=explanation.main_effects,
                                    hierarchical_values=explanation.hierarchical_values,
                                    clustering=explanation.clustering,
    )
    shap.plots.beeswarm(explanation_)

def get_ylabels(explanation_patient, X_patient): 
    Histology_it = 'Yes' if X_patient['Histology'].values == 'Intestinal type' else 'No'
    Histology_dt = 'Yes' if X_patient['Histology'].values == 'Diffuse type' else 'No'
    Location_l =  'Yes' if X_patient['Location'].values == 'Lower' else 'No'
    Location_m =  'Yes' if X_patient['Location'].values == 'Middle' else 'No'
    Location_u =  'Yes' if X_patient['Location'].values == 'Upper' else 'No'
    SRCC = 'Yes' if X_patient['SRCC'].values == True else 'No'
    
    ylabels = [
        str(X_patient['Tumor size'].values[0].astype(int)) + ' mm' + ' = ' + 'Tumor size',
        str(X_patient['T category, broad'].values[0]) + ' = ' + 'T category, broad',
        str(X_patient['T category'].values[0]) + ' = ' + 'T category',
        str(SRCC) + ' = ' + 'SRCC', 
        str(X_patient['Grade'].values[0]) + ' = ' + 'Grade',
        str(Location_l) + ' = ' + 'Location - Lower', 
        str(Location_m) + ' = ' + 'Location - Middle', 
        str(Location_u) + ' = ' + 'Location - Upper', 
        str(Histology_dt) + ' = ' + 'Histology - Diffuse type', 
        str(Histology_it) + ' = ' + 'Histology - Intestinal type',     
    ]
    combine_list = list(zip(
        np.abs(explanation_patient.values),
        explanation_patient.feature_names, 
        ylabels))
    sorted_lists = sorted(combine_list, key = lambda x: x[0], reverse = False)
    sorted_ylabels = [item[2] for item in sorted_lists]
   
    return sorted_ylabels

def plot_single_shap_waterfall(explanation, X_test, model, X_test_model, id):
    explanation_patient = explanation[id]
    X_patient = X_test.iloc[id:id+1]

    
    Probability = model.predict_proba(X_test_model.iloc[id:id+1])[0][1]
    Predicted = model.predict(X_test_model.iloc[id:id+1])
    Predicted_ = 'Positive' if Predicted == True else 'Negative'

    print(f'Predicted LNM: {Predicted_}; Probability of LNM: {Probability*100:.1f}%.')
    
    sorted_ylabels = get_ylabels(explanation_patient, X_patient)
    plt.clf()
    fig = shap.plots.waterfall(explanation_patient, max_display=18, show = False)
    ax_ = fig.get_axes()[0]
    tick_labels = ax_.yaxis.get_majorticklabels()
    for i in range(len(sorted_ylabels)):
        tick_labels[i].set_color("black")
    ax_.set_yticks(np.arange(len(sorted_ylabels)))
    ax_.set_yticklabels(sorted_ylabels)
    plot = ax_.get_figure()
    plot.set_size_inches(7.5,5)
    plot.savefig(f'No. {id} LNM {Predicted_} shap.pdf', format='pdf', bbox_inches = 'tight')
    plt.show()