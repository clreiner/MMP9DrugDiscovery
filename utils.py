
from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

def mannwhitney(DF, descriptor):
    # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
    # Based on the Data Professor's use of the above link
    
    '''
        Displays the Mann Whitney U Test statistic, p-value, and significance between 
        'active', 'intermediate', and 'inactive' classes
            Parameters:
                    descriptor: the feature to compare between classes
                    
    '''

    # Create samples by class
    selection = [descriptor, 'target']
    df = DF[selection]
    active = df[df['target'] == 'active']
    active = active[descriptor]

    selection = [descriptor, 'target']
    df = DF[selection]
    inactive = df[df['target'] == 'inactive']
    inactive = inactive[descriptor]

    selection = [descriptor, 'target']
    df = DF[selection]
    intermediate = df[df['target'] == 'intermediate']
    intermediate = intermediate[descriptor]

    # Compare samples using the Mann Whitney U test
    a_ia_stat, a_ia_p = mannwhitneyu(active, inactive)
    i_ia_stat, i_ia_p = mannwhitneyu(intermediate, inactive)
    a_i_stat, a_i_p = mannwhitneyu(active, intermediate)
    
    a_ia_prod = len(active)*len(inactive)
    i_ia_prod = len(inactive)*len(intermediate)
    a_i_prod = len(active)*len(intermediate)

    # Interpret the results
    alpha = 0.05
    interpretation = []
    for p in [a_ia_p, i_ia_p, a_i_p]:
        if p > alpha:
            interpretation.append('Same distribution (fail to reject H0)')
        else:
            interpretation.append('Different distribution (reject H0)')

        
    print(f'')
    print(f'{descriptor}: Active vs Inactive')
    print(f'Statistics: {a_ia_stat} of {a_ia_prod}, p: {a_ia_p: .5f}, Interpretation: {interpretation[0]}')
    print(f'{descriptor}: Intermediate vs Inactive')
    print(f'Statistics: {i_ia_stat} of {i_ia_prod}, p: {i_ia_p: .5f}, Interpretation: {interpretation[1]}')
    print(f'{descriptor}: Active vs Intermediate')
    print(f'Statistics: {a_i_stat} of {a_i_prod}, p: {a_i_p: .5f}, Interpretation: {interpretation[2]}')


def evaluate(model, X_train, X_test, y_train, y_test, use_decision_function='yes'):
    
    '''
        Displays the Train/Test accuracy, AUC/ROC score, precision, recall and F1 scores,
            and a confusion matrix
            Parameters:
                    model: a model fit to the training data
                    X_train (DataFrame): the training data
                    X_test (DataFrame): the test data
                    y_train (Series): the training target
                    y_test (Series): the test target
                    use_decision_function (string): default='yes', whether to use decision 
                                                function as not all model types can use;
    '''
    
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 5))
    
    train_pred = model.predict(X_train)
    cm_train = confusion_matrix(y_train, train_pred, labels=[0,1])
    disp_train = ConfusionMatrixDisplay(cm_train, display_labels=['inactive','active'])
    disp_train.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Train Matrix')
    
    
    test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, test_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=['inactive','active'])
    disp.plot(ax=ax2, cmap='Blues')
    ax2.set_title('Test Matrix')
    plt.show()
    
    if use_decision_function == 'skip': # skips calculating the roc_auc_score
        train_out = False
        test_out = False
    elif use_decision_function == 'yes': # not all classifiers have decision_function
        train_out = model.decision_function(X_train)
        test_out = model.decision_function(X_test)
    elif use_decision_function == 'no':
        train_out = model.predict_proba(X_train)[:, 1] # proba for the 1 class
        test_out = model.predict_proba(X_test)[:, 1]
    else:
        raise Exception ("The value for use_decision_function should be 'skip', 'yes' or 'no'.")
    
    train_acc = accuracy_score(y_train, train_pred) 
    train_roc_auc = roc_auc_score(y_train, train_out)
    train_precision, train_recall, train_fscore, support = precision_recall_fscore_support(y_train, 
                                                                                           train_pred, 
                                                                                           average='weighted')
    print('Train scores:')
    print(f'Accuracy: {train_acc: .3f}, ROC/AUC: {train_roc_auc: .3f}')
    print(f'Precision: {train_precision: .3f}, Recall: {train_recall: .3f}, F1 Score: {train_fscore: .3f}')

    test_acc = accuracy_score(y_test, test_pred) 
    test_roc_auc = roc_auc_score(y_test, test_out)
    test_precision, test_recall, test_fscore, support = precision_recall_fscore_support(y_test, 
                                                                                        test_pred, 
                                                                                        average='weighted')
    print('Test scores:')
    print(f'Accuracy: {test_acc: .3f}, ROC/AUC: {test_roc_auc: .3f}')
    print(f'Precision: {test_precision: .3f}, Recall: {test_recall: .3f}, F1 Score: {test_fscore: .3f}')