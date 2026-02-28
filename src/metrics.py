import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as sklearn_classification_report

def accuracy(y_true, y_pred):
    """Calcula el accuracy score."""
    return accuracy_score(y_true, y_pred)

def variance(po_pred, pt_pred, pg_pred):
    """Calcula la variabilidad (en qué % no coinciden las 3 outputs simultáneamente)."""
    # Convertir a numpy arrays por si vienen como listas (ej. desde un Dataset de HuggingFace)
    po = np.array(po_pred)
    pt = np.array(pt_pred)
    pg = np.array(pg_pred)
    
    all_match = (po == pt) & (po == pg)
    return (~all_match).mean()

def classification_report(y_true, y_pred, **kwargs):
    """Genera el reporte de clasificación."""
    return sklearn_classification_report(y_true, y_pred, **kwargs)
