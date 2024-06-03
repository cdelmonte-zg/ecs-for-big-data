import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple


def ecs_solution(transaction_data, user_profiles):
    class Entity:
        def __init__(self, entity_id):
            self.entity_id = entity_id
            self.components = {}

        def add_component(self, component_name, component):
            self.components[component_name] = component

        def get_component(self, component_name):
            return self.components.get(component_name)

    class TransactionComponent:
        def __init__(self, transaction_id, amount, timestamp, merchant_id):
            self.transaction_id = transaction_id
            self.amount = amount
            self.timestamp = timestamp
            self.merchant_id = merchant_id

    class UserProfileComponent:
        def __init__(self, user_id, user_history):
            self.user_id = user_id
            self.user_history = user_history

    class FeatureVectorComponent:
        def __init__(self, feature_vector):
            self.feature_vector = feature_vector

    class RiskScoreComponent:
        def __init__(self, risk_score):
            self.risk_score = risk_score

    def create_transaction_component(transaction_id, amount, timestamp, merchant_id):
        return TransactionComponent(transaction_id, amount, timestamp, merchant_id)

    def create_user_profile_component(user_id, user_history):
        return UserProfileComponent(user_id, user_history)

    def data_collection_system(transaction_data, user_profiles):
        entities = {}
        for t in transaction_data:
            entity_id = t[0]
            if entity_id not in entities:
                entities[entity_id] = Entity(entity_id)
            entities[entity_id].add_component('TransactionComponent', create_transaction_component(*t))

        for u in user_profiles:
            entity_id = u[0]
            if entity_id in entities:
                entities[entity_id].add_component('UserProfileComponent', create_user_profile_component(*u))

        return entities

    def data_preprocessing_system(entities):
        transactions = []
        user_profiles = {}
        for entity in entities.values():
            transaction = entity.get_component('TransactionComponent')
            user_profile = entity.get_component('UserProfileComponent')
            if transaction:
                transactions.append(transaction)
            if user_profile:
                user_profiles[entity.entity_id] = user_profile

        transaction_df = pd.DataFrame([(t.transaction_id, t.amount, t.timestamp, t.merchant_id) for t in transactions],
                                      columns=["transaction_id", "amount", "timestamp", "merchant_id"])
        transaction_df['amount_normalized'] = (transaction_df['amount'] - transaction_df['amount'].mean()) / transaction_df['amount'].std()

        for entity in entities.values():
            transaction = entity.get_component('TransactionComponent')
            if transaction:
                normalized_amount = transaction_df.loc[transaction_df['transaction_id'] == transaction.transaction_id, 'amount_normalized'].values[0]
                user_profile = user_profiles.get(entity.entity_id, UserProfileComponent(entity.entity_id, []))
                user_history_length = len(user_profile.user_history)
                feature_vector = [normalized_amount, user_history_length]
                entity.add_component('FeatureVectorComponent', FeatureVectorComponent(feature_vector))

    def transaction_analysis_system(entities):
        def analyze(features):
            return np.mean(features)  # Example: mean of features

        for entity in entities.values():
            feature_vector = entity.get_component('FeatureVectorComponent')
            if feature_vector:
                risk_score = analyze(feature_vector.feature_vector)
                entity.add_component('RiskScoreComponent', RiskScoreComponent(risk_score))

    def risk_prediction_system(model, entities):
        feature_vectors = []
        entity_ids = []
        for entity in entities.values():
            feature_vector = entity.get_component('FeatureVectorComponent')
            if feature_vector:
                feature_vectors.append(feature_vector.feature_vector)
                entity_ids.append(entity.entity_id)

        if feature_vectors:
            predictions = model.predict(feature_vectors)
            for entity_id, score in zip(entity_ids, predictions):
                entities[entity_id].add_component('RiskScoreComponent', RiskScoreComponent(score))

    def fraud_detection_system(entities, threshold=0.7):
        detected_frauds = []
        for entity in entities.values():
            risk_score = entity.get_component('RiskScoreComponent')
            if risk_score and risk_score.risk_score > threshold:
                detected_frauds.append((entity.entity_id, risk_score.risk_score))
        return detected_frauds

    # Pipeline execution
    model = RandomForestClassifier()
    X_train = np.array([[100, 1], [200, 2], [150, 1.5], [50, 0.5]])  # Dummy training data
    y_train = np.array([0, 1, 0, 1])  # Dummy labels
    model.fit(X_train, y_train)

    entities = data_collection_system(transaction_data, user_profiles)
    data_preprocessing_system(entities)
    transaction_analysis_system(entities)
    risk_prediction_system(model, entities)
    detected_frauds = fraud_detection_system(entities)

    return detected_frauds
