# tests/test_get_coef.py

import unittest
import pandas as pd
import numpy as np
from coef_analysis import get_coef

class TestGetCoef(unittest.TestCase):

    def setUp(self):
        # Criando um DataFrame de exemplo
        np.random.seed(42)
        self.df = pd.DataFrame({
            'A': np.random.rand(100),
            'B': np.random.rand(100) * 0.1,  # Baixa correlação com A
            'C': np.random.rand(100) * 0.5,  # Baixa correlação com A
            'Target': np.random.rand(100)    # Variável alvo
        })

    def test_get_coef(self):
        # Testando com limites baixos
        result = get_coef(self.df, 'Target', limite=[-0.2, 0.2])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Variável', result.columns)
        self.assertIn('Coeficiente de Correlação', result.columns)

    def test_no_correlation(self):
        # Deve levantar um erro se nenhuma correlação for encontrada
        with self.assertRaises(ValueError):
            get_coef(self.df, 'Target', limite=[-0.01, 0.01])

    def test_invalid_target(self):
        # Deve levantar um erro se a variável alvo não existir
        with self.assertRaises(ValueError):
            get_coef(self.df, 'Inexistente')

if __name__ == "__main__":
    unittest.main()
