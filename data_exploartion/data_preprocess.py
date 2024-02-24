from enum import Enum

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from data_exploartion.functions import get_score
ROUNDING_FACTOR = 3
seed = 43

class ProcessedData:
    def __init__(self):
        # Load the data
        self.data = pd.read_excel("../data/Date_Fruit_Datasets.xlsx")

        # add a column named "NULL" with all 0:
        self.data['NULL'] = 0
        self.null_col_index = len(self.data.columns) - 1

        self.train_set, self.test_set = train_test_split(self.data, test_size=0.35, random_state=seed, stratify=self.data['Class'])

        self.y_train = self.train_set['Class']
        self.X_train = self.train_set.drop('Class', axis=1)

        self.y_test = self.test_set['Class']
        self.X_test = self.test_set.drop('Class', axis=1)

        self.feature_importance_df = None


    def train_decision_tree(self):
        # Train a decision tree classifier
        self.dt = DecisionTreeClassifier(random_state=seed)
        self.dt.fit(self.X_train, self.y_train)

        # Extract feature importances
        feature_importance = self.dt.feature_importances_

        feature_names = self.data.columns  # Replace this with your actual feature names
        feature_names = feature_names.drop('Class')

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance':feature_importance
        })

        # sort by importance
        self.feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index()

    def train_logistic_regressors(self, ranking_measure):

        importance_table = pd.DataFrame()

        features_to_include = 34
        if features_to_include > len(self.X_train.columns):
            features_to_include = len(self.X_train.columns)

        if ranking_measure == "tree":
            self.train_decision_tree()
            importance_table = self.feature_importance_df.copy().head(features_to_include).reset_index(
                drop=True)
            importance_table = importance_table.drop('index', axis=1)

        # Prepare stratified K-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        final_df = pd.DataFrame()

        selected_features = []
        scaler = MinMaxScaler()
        scaler.fit(self.X_train)
        X_selected_standardized = scaler.transform(self.X_train)

        for i in tqdm(range(0, features_to_include+1)):
            model = LogisticRegression(random_state=seed,
                                       max_iter=10000,
                                       multi_class='multinomial',
                                       solver='lbfgs',
                                       )

            all_features = list(self.X_train.columns)
            features_to_check = []

            if i == 0:
                # Fit the null model
                features_to_check = ['NULL']
            else:
                # Remove the last feature from the selected features which is all 0 for the null model
                all_features = all_features[:-1]
                if ranking_measure == "tree":
                    features_to_check = [importance_table['Feature'][i-1]]
                if ranking_measure == "BIC" or ranking_measure == "train_accuracy":
                    # add all features that are not in selected_features from all features
                    features_to_check = []
                    for feature in all_features:
                        if feature not in selected_features:
                            features_to_check.append(feature)

            temp_df = pd.DataFrame()

            for feature in features_to_check:
                current_num_of_features = len(selected_features) + 1
                new_row = {"total_features":current_num_of_features, 'added_features': feature}

                if ranking_measure == 'tree':
                    if feature == "NULL":
                        new_row.update({'importance': 0})
                    else:
                        # if we do tree - add feature importance
                        new_row.update({'importance':
                                            round(importance_table[importance_table['Feature'] == feature]['Importance'].values[
                                                0],ROUNDING_FACTOR)})

                # Get all features to check
                to_check = selected_features.copy()

                # Add the current feature
                to_check.append(feature)

                # Get the indices of the features including the new one
                selected_features_indicis = [all_features.index(feature) for feature in to_check]

                # Get x_train
                x_train = X_selected_standardized[:, selected_features_indicis]


                # Perform 5-fold stratified cross-validation and calculate metrics
                scoring = cross_validate(model, x_train, self.y_train, scoring=get_score, cv=cv, n_jobs=1)

                # Calculate average of the scores
                for score_title in scoring.keys():
                        new_row[f"{score_title}_mean"] = round(scoring[score_title].mean(),ROUNDING_FACTOR)

                # Append to the DataFrame
                temp_df = temp_df._append(new_row, ignore_index=True)


            if ranking_measure == "BIC":
                temp_df = temp_df.sort_values(by='test_BIC_mean', ascending=True)
            if ranking_measure == "train_accuracy":
                temp_df = temp_df.sort_values(by='test_accuracy_mean', ascending=False)

            new_row = temp_df.head(1)
            if new_row['added_features'].values[0] != "NULL":
                selected_features.append(new_row['added_features'].values[0])
            else:
                selected_features = []
                X_selected_standardized = X_selected_standardized[:, :-1]

            final_df = final_df._append(new_row, ignore_index=True)

        final_df = final_df.rename(columns={'test_log_likelihood_mean': 'log_likelihood', 'test_accuracy_mean': 'accuracy'})

        final_df['accuracy_difference'] = round(final_df['accuracy'].diff().fillna(0), ROUNDING_FACTOR)
        final_df['log_likelihood_ratio'] = round(2*final_df['log_likelihood'].diff().fillna(0), ROUNDING_FACTOR)

        # Save all metrics to a single CSV
        final_df.drop(['fit_time_mean', 'score_time_mean'], axis=1, inplace=True)
        final_df.to_csv(f'model_metrics_summary_{ranking_measure}.csv', index=False)

    def plot_histogram_of_class_feature_in_date(self):
        # Plot the histogram of the Class feature
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        categories = ['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY']
        sns.histplot(self.data['Class'], bins=len(categories), palette="viridis", edgecolor='black')
        plt.title('Histogram of the Class Feature', fontsize=16, fontname="Sans")
        plt.xlabel('Class', fontsize=14, fontname="Sans")
        plt.ylabel('Frequency', fontsize=14, fontname="Sans")
        plt.xticks(fontsize=12, fontname="Sans")
        plt.yticks(fontsize=12, fontname="Sans")
        plt.tight_layout()
        plt.savefig('histogram_of_class_feature_beautiful.png', dpi=300)
        plt.show()

    def test_logistic_regression(self, final_features):
        all_features = list(self.data.columns)
        confusion_matrices = []
        results = []
        # Train a logistic regression model
        for data_type in ['train', 'test', 'combined']:
            X = self.X_train
            y = self.y_train

            if data_type == 'train':
                X_test = self.X_train
                y_test = self.y_train
            elif data_type == 'test':
                X_test = self.X_test
                y_test = self.y_test
            else:
                X_test = pd.concat([self.X_train, self.X_test])
                y_test = pd.concat([self.y_train, self.y_test])

            # Remove NULL
            X = X.drop('NULL', axis=1)
            X_test = X_test.drop('NULL', axis=1)

            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X_test = scaler.transform(X_test)

            # Initialize the logistic regression model
            model = LogisticRegression(random_state=seed,
                                       max_iter=10000,
                                       multi_class='multinomial',
                                       solver='lbfgs',
                                       penalty=None)

            final_features_indicis = [all_features.index(feature) for feature in final_features]

            # Fit the model
            X = X[:,final_features_indicis]
            X_test = X_test[:,final_features_indicis]
            model.fit(X, y)

            coefs = model.coef_


            # Get the predictions
            y_pred = model.predict(X_test)

            # Get the accuracy
            accuracy = np.mean(y_test == y_pred)
            recall = np.mean(y_test == y_pred)
            precision = np.mean(y_test == y_pred)
            f1 = 2 * (precision * recall) / (precision + recall)
            results.append({'data':data_type,
                            'accuracy': round(accuracy,ROUNDING_FACTOR),
                            'recall': round(recall,ROUNDING_FACTOR),
                            'precision': round(precision,ROUNDING_FACTOR),
                            'f1': round(f1,ROUNDING_FACTOR),})

            confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
            confusion_matrices.append(confusion_matrix)
        # save results to a csv
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'logistic_regression_results.csv', index=False)

        # save confusion matrices to a csv
        confusion_matrices_df = pd.concat(confusion_matrices, keys=['train', 'test', 'combined'])
        confusion_matrices_df.to_csv(f'logistic_regression_confusion_matrices.csv')

    def save_plot_correlation_matrix(self, features=None):
        # Plot the correlation matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        all_features_names = list(self.data.columns.drop(['Class', "NULL"]))
        X = self.data.drop('Class', axis=1)
        X = X.to_numpy()
        if features is None:
            feature_names = self.data.columns.drop('Class')
        else:
            feature_names = features
        X_without_null = X[:, :-1]

        features_indicis = []
        for i in range(len(feature_names)):
            feature = feature_names[i]
            features_indicis.append(all_features_names.index(feature))

        X_only_for_features = X_without_null[:,features_indicis]

        corr = np.corrcoef(X_only_for_features, rowvar=False)

        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                plt.text(i, j, round(corr[i, j], 2), ha='center', va='center', color='black')

        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names)
        ax.set_yticklabels(feature_names)
        cax = ax.imshow(corr, cmap='coolwarm', interpolation='none')
        cbar = fig.colorbar(cax)
        cbar.set_label('Correlation',)

        # shrink color bar to be the same size as table

        cbar.ax.set_ylabel('Correlation Coefficient', rotation=-90, va="bottom")
        plt.tight_layout()


        plt.title("Correlation Matrix")
        plt.savefig("correlation_matrix.png")


    ## Visualizations



    def save_conditional_distributions(self, df):
        # Ensure the folder exists
        folder_path = "conditional_distributions"
        target_column = "Class"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Find unique conditions
        conditions = df[target_column].unique()

        # Exclude the target column from the features list
        features = df.columns.difference([target_column])

        for feature in features:
            plt.figure()

            # Plot normalized distribution for each condition on the same plot
            for condition in conditions:
                subset = df[df[target_column] == condition]
                subset[feature].hist(alpha=0.5, label=f"Condition {condition}", density=True)

            plt.title(f"Normalized Distributions of {feature} by Condition")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend()

            # Save the plot
            filename = f"{folder_path}/{feature}_normalized_conditions.png"
            plt.savefig(filename)
            plt.close()  # Close the plot to free up memory

    def plot_pca_2d(self, cols):
        df = self.data
        # Ensure "Class" is part of the DataFrame and filter the DataFrame to use only the specified columns and "Class"
        if 'Class' not in df.columns:
            raise ValueError('"Class" column must be present in the DataFrame.')

        X = df[cols]
        y = 'Class'

        # Standardize the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        print("PCA Explained variance ratio:", pca.explained_variance_ratio_)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(X_scaled)

        # Plotting PCA
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        self.plot_result(pca_result, df['Class'].values, 'PCA')

        # Plotting t-SNE
        plt.subplot(1, 2, 2)
        self.plot_result(tsne_result, df['Class'].values, 't-SNE')
        plt.show()

    def plot_result(self, result, classes, title):
        df_result = pd.DataFrame(data=result, columns=['Component 1', 'Component 2'])
        df_result['Class'] = classes

        ax = plt.gca()
        targets = np.unique(classes)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(targets)))

        for target, color in zip(targets, colors):
            indicesToKeep = df_result['Class'] == target
            ax.scatter(df_result.loc[indicesToKeep, 'Component 1'],
                       df_result.loc[indicesToKeep, 'Component 2'],
                       c=[color],
                       s=50,
                       label=target)
        plt.title(title)
        plt.legend(targets)
        plt.grid()


class Options(Enum):
    BIC = "BIC"
    tree = "tree"
    train_accuracy = "train_accuracy"



final_feat = ['MINOR_AXIS', "SkewRB", "SkewRG", 'MeanRR', 'ALLdaub4RR', 'PERIMETER', 'SkewRR', 'StdDevRG']
less_cor = ['MINOR_AXIS', "SkewRB", 'MeanRR', 'StdDevRG']
data = ProcessedData()


data_set = data.data
data_set = data_set.drop(['NULL', 'Class'], axis=1)
all_features = list(data_set.columns)


data.plot_histogram_of_class_feature_in_date()
# data.save_plot_correlation_matrix(features=all_features)
# # data.plot_pca_2d(all_features)
# data.train_logistic_regressors(Options.train_accuracy.value)
# data.train_logistic_regressors(Options.BIC.value)
# data.train_logistic_regressors(Options.tree.value)
# data.test_logistic_regression(less_cor)