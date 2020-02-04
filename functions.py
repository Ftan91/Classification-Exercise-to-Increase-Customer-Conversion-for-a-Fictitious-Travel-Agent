import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def print_classification_summary(model_name, dataset, model_instance, y, X, positive_label = 1, negative_label=0):
    """Function to outlay summary information of chosen model including target distribution for dataset used, classification metric scores, confusion matrix and ROC/AUC and Precision/Recall curves."""
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix 
    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve 
    
    y_pred = model_instance.predict(X)
    class_y = np.unique(y)
    
    print ('***RESULTS SUMMARY***')    
    print ('------------------------------------------------------------------------------------------')
    print ('Model:', model_name)
    print ('Dataset:', dataset)
    print ('Target distribution:')
    print ('\n')
    print ('Class:', class_y[0], '/ Count:', sum(y == class_y[0]), '/ Pct:', round(sum(y == class_y[0]) / len(y) * 100,0))
    print ('Class:', class_y[1], '/ Count:', sum(y == class_y[1]), '/ Pct:', round(sum(y == class_y[1]) / len(y) * 100,0))
    print ('------------------------------------------------------------------------------------------')
    print ('Metric Scores: \n')
    print ('Accuracy score:', round(accuracy_score(y, y_pred),2))
    print ('Recall score:', round(recall_score(y, y_pred), 2))
    print ('Precision score:', round(precision_score(y, y_pred), 2))
    print ('F1 score:', round(f1_score(y, y_pred), 2))
    print ('------------------------------------------------------------------------------------------')
    print ('Plots:')
    ax = plot_confusion_matrix(model_instance, X, y,values_format='d')
    plt.title('Confusion Matrix')
    plt.show()
    ax = plot_roc_curve(model_instance, X, y)
    plt.title('ROC Curve')
    plt.show()
    ax = plot_precision_recall_curve(model_instance, X, y)
    plt.title('Precision Recall Curve')
    plt.show()
    
def print_feature_importance(taxonomy,model,columns,top_selector=10):
    """Function to Plot the Top Features of an ensemble model"""
    taxonomy['last_two'] = taxonomy.Label.str.split('§')
    taxonomy['last_two'] = taxonomy['last_two'].apply(lambda x: x[-2] + '/' + x[-1])

    rf_feature = pd.DataFrame(model.feature_importances_, index = columns, columns=['Importance'])
    rf_feature_importance_df = rf_feature.sort_values(by = 'Importance', ascending = False)
    subset = rf_feature_importance_df.iloc[:top_selector]*100
    subset.plot(kind = 'barh')
    plt.title('Top {} Features'.format(top_selector))
    labels = taxonomy.loc[subset.index.str.replace("_A", "").astype('int64')]['last_two']
    plt.ylabel('Options')
    plt.xlabel('Importance %')
    plt.legend([])
    plt.yticks(range(10), labels.values)
    plt.show()

    
def remove_p(df):
    """Function to remove _P from tags"""
    new_col = []
    for column in df.columns:
        if column.endswith('_P'):
            continue
        else:
            new_col.append(column)
    cleaned_data = df[new_col]
    return cleaned_data

def make_radar_chart(name, stats, attribute_labels):
    """Function to make a spider / radar chart from output derived from clustering exercise"""
    markers = [0.2, 0.4, 0.6, 0.8, 1]
    str_markers = ['0.2', '0.4', '0.6', '0.8', '1']
    labels = np.array(attribute_labels)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))
    fig= plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    labels = []
    plt.yticks(markers)
    ax.set_title(name)
    ax.grid(True)
    return plt.show()

def annot(fpr,tpr,thr):
    """Function to create annotations in the roc plot"""
    k=0
    for i,j in zip(fpr,tpr):
        if k%2 == 0:
            plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
        k+=1

def roc_plot(y_val,y_pred):
    """Function to plot roc curve with annotations"""
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(7,7))
    fpr, tpr, threshold = roc_curve(y_val,y_pred)
    plt.plot(fpr, tpr)
    annot(fpr, tpr, threshold)
    plt.ylabel('TPR (power)')
    plt.xlabel('FPR (alpha)')
    return plt.show()