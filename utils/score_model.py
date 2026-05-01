from autogluon.tabular import TabularPredictor

class ScoreModel:
    def __init__(self):
        self.predictor = None

    def train(self, df):
        self.predictor = TabularPredictor(label='score').fit(df)

    def predict(self, df):
        return self.predictor.predict(df)