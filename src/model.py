from xgboost import XGBClassifier

def get_model():
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss"
    )
    return model