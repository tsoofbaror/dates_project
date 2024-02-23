import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functions import get_score, get_next_feature_with_BIC


class ProcessedData:
    def __init__(self):
        # Load the data
        self.data = pd.read_excel("../data/Date_Fruit_Datasets.xlsx")

        self.train_set, self.test_set = train_test_split(self.data, test_size=0.15, random_state=42, stratify=self.data['Class'])

        self.y_train = self.train_set['Class']
        self.X_train = self.train_set.drop('Class', axis=1)

        self.y_test = self.test_set['Class']
        self.X_test = self.test_set.drop('Class', axis=1)

        self.feature_importance_df = None


    def train_decision_tree(self):
        # Train a decision tree classifier
        self.dt = DecisionTreeClassifier(random_state=41)
        self.dt.fit(self.X_train, self.y_train)

        # Extract feature importances
        feature_importance = self.dt.feature_importances_

        feature_names = self.data.columns  # Replace this with your actual feature names
        feature_names = feature_names.drop('Class')

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })

        # sort by importance
        self.feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index()

    def train_logistic_regressors(self, ranking_measure):
        self.train_decision_tree()  # Assuming this populates self.feature_importance_df

        # Prepare stratified K-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Loop through the top 1 to 6 features
        features_to_include = 34
        final_df = pd.DataFrame()

        importance_table = self.feature_importance_df.copy().head(features_to_include).reset_index(drop=True)
        null_model_row = pd.DataFrame({'Feature': ['Null Model'], 'Importance': [0]})
        importance_table = pd.concat([null_model_row, importance_table], axis=0, ignore_index=True)
        importance_table = importance_table.drop('index', axis=1)

        selected_features = []
        for i in range(0, features_to_include+1):
            # Initialize Logistic Regression model
            model = LogisticRegression(random_state=42, max_iter=10000, multi_class='multinomial', solver='lbfgs')
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            X_selected_standardized = scaler.transform(self.X_train)

            if i == 0:
                X_selected_standardized = np.array([0] * len(self.X_train)).reshape(-1, 1)
            else:
                if ranking_measure == "tree":
                    selected_features.append(importance_table['Feature'][i-1])
                if ranking_measure == 'BIC':
                    selected_features = get_next_feature_with_BIC(model, X_selected_standardized, self.y_train, list(importance_table['Feature']), cv)


                X_selected_standardized = self.X_train[selected_features]

                # Standardize features


            # Perform 5-fold stratified cross-validation and calculate metrics
            scoring = cross_validate(model, X_selected_standardized, self.y_train, scoring=get_score, cv=cv, n_jobs=1)
            new_row = importance_table.iloc[i].copy()
            # Calculate average of the scores
            for score_title in scoring.keys():
                    new_row[f"{score_title}_mean"] = round(scoring[score_title].mean(),3)

            new_row['num_features'] = i  # Include the number of features in the metrics

            # Append to the DataFrame
            final_df = final_df._append(new_row, ignore_index=True)


        final_df = final_df.rename(columns={'test_log_likelihood_mean': 'log_likelihood', 'test_accuracy_mean': 'accuracy'})

        final_df['accuracy_difference'] = final_df['accuracy'].diff().fillna(0)
        final_df['log_likelihood_ratio'] = 2*final_df['log_likelihood'].diff().fillna(0)

        # Save all metrics to a single CSV
        final_df.drop(['fit_time_mean', 'score_time_mean'], axis=1, inplace=True)
        final_df.to_csv('model_metrics_summary.csv', index=False)



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


data = ProcessedData()
data.train_logistic_regressors("BIC")
pass
