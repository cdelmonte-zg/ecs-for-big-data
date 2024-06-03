import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple

def ecs_functional_solution(transaction_data, user_profiles):
    Transaction = Tuple[int, int, str, str]
    UserProfile = Tuple[int, List[int]]
    FeatureVector = List[float]
    RiskScore = float

    def create_transaction(transaction_id: int, amount: int, timestamp: str, merchant_id: str) -> Transaction:
        return (transaction_id, amount, timestamp, merchant_id)

    def create_user_profile(user_id: int, user_history: List[int]) -> UserProfile:
        return (user_id, user_history)

    def create_feature_vector(features: List[float]) -> FeatureVector:
        return features

    def create_risk_score(score: float) -> RiskScore:
        return score

    def data_collection_system(transaction_data: List[Transaction], user_profiles: List[UserProfile]) -> Tuple[List[Transaction], List[UserProfile]]:
        return transaction_data, user_profiles

    def data_preprocessing_system(transactions: List[Transaction], user_profiles: List[UserProfile]) -> np.ndarray:
        transaction_df = pd.DataFrame(transactions, columns=["transaction_id", "amount", "timestamp", "merchant_id"])
        transaction_df['amount_normalized'] = (transaction_df['amount'] - transaction_df['amount'].mean()) / transaction_df['amount'].std()

        user_profiles_dict = {profile[0]: profile[1] for profile in user_profiles}

        feature_vectors = np.zeros((len(transactions), 2))

        for i, transaction in enumerate(transactions):
            transaction_id = transaction[0]
            normalized_amount = transaction_df.loc[transaction_df['transaction_id'] == transaction_id, 'amount_normalized'].values[0]
            user_history_length = len(user_profiles_dict.get(transaction_id, []))
            feature_vectors[i] = [normalized_amount, user_history_length]

        return feature_vectors

    def transaction_analysis_system(feature_vectors: np.ndarray) -> np.ndarray:
        return np.mean(feature_vectors, axis=1)

    def risk_prediction_system(model, feature_vectors: np.ndarray) -> np.ndarray:
        return model.predict(feature_vectors)

    def fraud_detection_system(risk_scores: np.ndarray, threshold: float = 0.7) -> List[Tuple[int, RiskScore]]:
        detected_frauds = [(i + 1, score) for i, score in enumerate(risk_scores) if score > threshold]
        return detected_frauds

    def execute_pipeline(transaction_data: List[Transaction], user_profiles: List[UserProfile], model) -> List[Tuple[int, RiskScore]]:
        transactions, profiles = data_collection_system(transaction_data, user_profiles)
        feature_vectors = data_preprocessing_system(transactions, profiles)
        preliminary_risk_scores = transaction_analysis_system(feature_vectors)
        risk_scores = risk_prediction_system(model, feature_vectors)
        detected_frauds = fraud_detection_system(risk_scores)
        return detected_frauds

    model = RandomForestClassifier()
    X_train = np.array([[100, 1], [200, 2], [150, 1.5], [50, 0.5]])
    y_train = np.array([0, 1, 0, 1])
    model.fit(X_train, y_train)

    detected_frauds = execute_pipeline(transaction_data, user_profiles, model)

    return detected_frauds
