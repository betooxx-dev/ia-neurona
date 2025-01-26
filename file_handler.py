import pandas as pd
from typing import Tuple, List, Dict
import numpy as np

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lee datos desde CSV o XLSX con múltiples variables X
    Retorna: X (array con todas las variables independientes), Y (variable dependiente)
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, sep=';')  # Usar ; como separador
    elif file_path.endswith(('.xlsx', '.xls')):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Formato de archivo no soportado. Use CSV o XLSX")
    
    # Identificar columnas X y Y
    y_column = 'Y'
    x_columns = [col for col in data.columns if col.startswith('X')]
    
    if not x_columns or y_column not in data.columns:
        raise ValueError("El archivo debe tener columnas X1,X2,... y Y")
    
    print(f"Columnas X: {x_columns}, Columna Y: {y_column}")
        
    return data[x_columns].values, data[y_column].values