import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import click
import os
import altair as alt

@click.command()
@click.option('--x-training-data', type=str, help="Path to x-train")
@click.option('--y-training-data', type=str, help="Path to y-train")
@click.option('--x-test-data', type=str, help="Path to x-test")
@click.option('--y-test-data', type=str, help="Path to y-test")

def main(x_train, y_train, x_test, y_test):
    preprocessor = OneHotEncoder(handle_unknown="ignore")

    dc_pipe = make_pipeline(
        preprocessor, 
        DummyClassifier(random_state=123)
    )

    dc_pipe.fit(x_train, y_train)

    cross_val_dc = pd.DataFrame(
        cross_validate(
            dc_pipe, x_train, y_train, cv=10, n_jobs=-1, return_train_score=True
        )).mean().to_frame().rename(columns={0: "mean_value"})

    svc_pipe = make_pipeline(
        preprocessor, 
        SVC(random_state=123)
    )

    svc_pipe.fit(x_train, y_train)

    cross_val_svc = pd.DataFrame(
        cross_validate(
            svc_pipe, x_train, y_train, cv=10, n_jobs=-1, return_train_score=True
        )).mean().to_frame().rename(columns={0: "mean_value"})


    y_pred = svc_pipe.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    plot = ConfusionMatrixDisplay(cm).plot()
    plot.save(os.path.join('../img/confusion_matrix.png', "cancer_choose_k.png"), scale_factor=2.0)


if __name__ == '__main__':
    main()