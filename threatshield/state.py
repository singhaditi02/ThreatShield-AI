"""
state.py  ·  Section 2 of the notebook (global application state)
One shared dictionary `S` holds the dataset, scaler, trained models and results.
Every module imports and mutates this same dictionary.
"""
S = dict(df=None, raw_train=None, raw_test=None, y_train=None, y_test=None,
         X_train=None, X_test=None, scaler=None, models={}, results={}, source="—")
