"""
Pacote de alta precisão (high precision) — 50 dígitos via mpmath.
Reimplementação independente do programa do Capítulo 4 da tese de R. Barros.
"""
from mpmath import mp
mp.dps = 50  # 50 dígitos de precisão decimal (≥32 casas garantidas)
