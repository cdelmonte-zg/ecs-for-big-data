
import random
from time import time

import numpy as np
import pandas as pd
import random
from time import time
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple

# Codice per la versione Visitor
def visitor_solution(transaction_data, user_profiles):
    class Entity:
        def __init__(self, id):
            self.id = id
            self.components = {}
        
        def add_component(self, component):
            self.components[type(component)] = component
            component.entity = self
        
        def get_component(self, component_type):
            return self.components.get(component_type, None)

    class TransactionComponent:
        def __init__(self, transaction_id, amount, timestamp, merchant_id):
            self.transaction_id = transaction_id
            self.amount = amount
            self.timestamp = timestamp
            self.merchant_id = None  # Corretto per coerenza
            self.entity = None
    
        def accept(self, visitor):
            visitor.visit_transaction(self)

    class UserProfileComponent:
        def __init__(self, user_id, user_history):
            self.user_id = user_id
            self.user_history = user_history
            self.entity = None
    
        def accept(self, visitor):
            visitor.visit_user_profile(self)

    class FeatureVectorComponent:
        def __init__(self, features):
            self.features = features
            self.entity = None
    
        def accept(self, visitor):
            visitor.visit_feature_vector(self)

    class RiskScoreComponent:
        def __init__(self, score):
            self.score = score
            self.entity = None
    
        def accept(self, visitor):
            visitor.visit_risk_score(self)

    class Visitor:
        def visit_transaction(self, component):
            pass

        def visit_user_profile(self, component):
            pass

        def visit_feature_vector(self, component):
            pass

        def visit_risk_score(self, component):
            pass

    class FeatureVectorVisitor(Visitor):
        def __init__(self, transaction_df, user_profiles):
            self.transaction_df = transaction_df
            self.user_profiles = user_profiles

        def visit_transaction(self, component):
            normalized_amount = self.transaction_df.loc[
                self.transaction_df['transaction_id'] == component.transaction_id, 
                'amount_normalized'
            ].values[0]
            user_history_length = len(self.user_profiles.get(component.entity.id, UserProfileComponent(component.entity.id, [])).user_history)
            component.entity.add_component(FeatureVectorComponent([normalized_amount, user_history_length]))

    class RiskScoreVisitor(Visitor):
        def visit_feature_vector(self, component):
            score = np.mean(component.features)
            component.entity.add_component(RiskScoreComponent(score))

    class DataCollectionSystem:
        def update(self, entities):
            for t in transaction_data:
                entity = Entity(t[0])
                transaction_component = TransactionComponent(*t)
                transaction_component.entity = entity
                entity.add_component(transaction_component)
                entities.append(entity)

            for u in user_profiles:
                entity = next((e for e in entities if e.id == u[0]), None)
                if entity:
                    user_profile_component = UserProfileComponent(*u)
                    user_profile_component.entity = entity
                    entity.add_component(user_profile_component)

    class DataPreprocessingSystem:
        def update(self, entities):
            transactions = []
            user_profiles = {}
            for entity in entities:
                transaction = entity.get_component(TransactionComponent)
                user_profile = entity.get_component(UserProfileComponent)
                if transaction and user_profile:
                    transactions.append(transaction)
                    user_profiles[entity.id] = user_profile

            transaction_df = pd.DataFrame([vars(t) for t in transactions])
            transaction_df['amount_normalized'] = (transaction_df['amount'] - transaction_df['amount'].mean()) / transaction_df['amount'].std()

            visitor = FeatureVectorVisitor(transaction_df, user_profiles)
            for entity in entities:
                transaction = entity.get_component(TransactionComponent)
                if transaction:
                    transaction.accept(visitor)

    class TransactionAnalysisSystem:
        def update(self, entities):
            visitor = RiskScoreVisitor()
            for entity in entities:
                features = entity.get_component(FeatureVectorComponent)
                if features:
                    features.accept(visitor)

    class RiskPredictionSystem:
        def __init__(self, model):
            self.model = model

        def update(self, entities):
            feature_vectors = []
            for entity in entities:
                features = entity.get_component(FeatureVectorComponent)
                if features:
                    feature_vectors.append(features.features)

            if feature_vectors:
                predictions = self.model.predict(feature_vectors)
                for entity, score in zip(entities, predictions):
                    entity.add_component(RiskScoreComponent(score))

    class FraudDetectionSystem:
        def update(self, entities):
            detected_frauds = []
            for entity in entities:
                risk_score = entity.get_component(RiskScoreComponent)
                if risk_score and risk_score.score > 0.7:  # soglia di esempio
                    detected_frauds.append((entity.id, risk_score.score))
            return detected_frauds

    # Esecuzione della pipeline
    model = RandomForestClassifier()
    X_train = np.array([[100, 1], [200, 2], [150, 1.5], [50, 0.5]])  # Dati di addestramento fittizi
    y_train = np.array([0, 1, 0, 1])  # Etichette di addestramento fittizie
    model.fit(X_train, y_train)

    entities = []

    data_collection_system = DataCollectionSystem()
    data_preprocessing_system = DataPreprocessingSystem()
    transaction_analysis_system = TransactionAnalysisSystem()
    risk_prediction_system = RiskPredictionSystem(model)
    fraud_detection_system = FraudDetectionSystem()

    data_collection_system.update(entities)
    data_preprocessing_system.update(entities)
    transaction_analysis_system.update(entities)
    risk_prediction_system.update(entities)
    detected_frauds = fraud_detection_system.update(entities)
    
    return detected_frauds
