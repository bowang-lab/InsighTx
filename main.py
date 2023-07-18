import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from evaluate import cross_val_score, get_shaps


class EVLP_XGBoost:
    def __init__(self, model_path: str | None = None):
        self.trained_model = (
            None if model_path is None else pickle.load(open(model_path, "rb"))
        )

    def _get_relevant_input(
        self,
        donor_data,
        evlp_data,
    ):
        ## Load data
        evlp_donor_data = pd.merge(
            donor_data,
            evlp_data,
            left_on="EVLP_number",
            right_on="EVLP number",
            how="left",
        )
        evlp_donor_data.drop(["EVLP number"], axis="columns", inplace=True)
        X_df, y_df = ...

        return X_df, y_df

    def __call__(
        self,
        donor_data_path: str,
        evlp_data_path: str,
        result_save_path: str,
    ):
        if self.trained_model is None:
            raise ValueError("Model not trained yet! Must first call train() method.")

        warnings.filterwarnings("ignore")

        evlp_data = pd.read_csv(evlp_data_path)
        donor_data = pd.read_csv(donor_data_path)
        X_df, y_df = self._get_relevant_input(
            donor_data,
            evlp_data,
        )

        print("\nTesting results ...")

        lb = OneHotEncoder()
        lb.fit(y_df.values.reshape([-1, 1]))
        y_df_binarized = lb.transform(y_df.values.reshape([-1, 1])).toarray()

        ## predict the held-out data
        held_out_predict = self.trained_model.predict(X_df)
        held_out_predict_proba = self.trained_model.predict_proba(X_df)

        results = {
            "Predicted Label": held_out_predict,
            "Predicted Probability (Tx, <72h)": held_out_predict_proba[:, 0],
            "Predicted Probability (Tx, >=72h)": held_out_predict_proba[:, 1],
            "Predicted Probability (Declined)": held_out_predict_proba[:, 2],
        }

        test_results_df = pd.DataFrame(results).to_csv(
            result_save_path + "test_results.csv", index=False
        )
        print(
            "Overall ROC score: ",
            roc_auc_score(
                y_df_binarized,
                held_out_predict_proba,
                multi_class="ovr",
                average="macro",
            ),
        )
        print(
            "ROC per class: ",
            roc_auc_score(
                y_df_binarized, held_out_predict_proba, multi_class="ovr", average=None
            ),
        )

        get_shaps(X_df, self.trained_model, result_save_path)

        print("#################################################")
        warnings.filterwarnings("default")

        return test_results_df

    def train(
        self,
        donor_data_path: str,
        evlp_data_path: str,
        generate_split: bool,
        split_path: str,
        model_name: str,
        param_grid: dict,
    ):
        select_k_best_list = np.arange(250, 251)
        ## initiaalize a dict to save the model parameters
        all_best_model_params = {}

        warnings.filterwarnings("ignore")

        for select_k_best in select_k_best_list:
            evlp_data = pd.read_csv(evlp_data_path)
            donor_data = pd.read_csv(donor_data_path)
            X_df, y_df = self._get_relevant_input(donor_data, evlp_data)

            print("Number of all features in the dataset: ", len(X_df.columns))
            print("List of all features: \n", X_df.columns)

            ## load the kfold indices
            if generate_split:
                rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
                splits = []

                for train_index, test_index in rskf.split(X_df, y_df):
                    splits.append((train_index, test_index))

                file = open(split_path, "wb")
                pickle.dump(splits, file)

            file = open(split_path, "rb")
            data_splits = pickle.load(file)

            ## set the XGBoost pipeline
            print("Set the pipeline for multiclass classification ...")
            model_classifier = xgb.XGBClassifier(...)
            steps = [("model", model_classifier)]

            ## compute class weights
            class_weights = class_weight.compute_class_weight(
                "balanced", np.unique(y_df.values), y_df.values
            )
            print("Class weights:", class_weights)
            instance_weights = [class_weights[int(i)] for i in y_df.values]
            instance_weights = np.array(instance_weights)
            instance_weights_df = pd.DataFrame(instance_weights, columns=["Weights"])

            ## grid search
            full_pipeline_EVLP = Pipeline(steps=steps)
            param_grid = param_grid
            gsearch = GridSearchCV(
                full_pipeline_EVLP,
                param_grid=param_grid,
                scoring=["roc_auc_ovr"],
                cv=data_splits,
                refit="roc_auc_ovr",
                return_train_score=True,
            )
            gsearch.fit(X_df, y_df, model__sample_weight=instance_weights)

            ## save the best parameters of model from gsearch
            all_best_model_params[select_k_best] = gsearch.best_params_
            best_model = gsearch.best_estimator_

            cross_val_score(best_model, X_df, y_df, data_splits, instance_weights_df)

            warnings.filterwarnings("default")

            ## save the best model
            pickle.dump(best_model, open(model_name, "wb"))
            self.trained_model = best_model


# To train the model, set the model_path to None
model = EVLP_XGBoost(model_path=None)
model.train(
    donor_data_path=...,
    evlp_data_path=...,
    generate_split=True,
    split_path=...,
    model_name=...,
    param_grid=...,
)
# To perform inference, pass in model_path in the EVLP_XGBoost class, and run the predict function
prediction = model(
    donor_data_path=...,
    evlp_data_path=...,
    result_save_path=...,
)
