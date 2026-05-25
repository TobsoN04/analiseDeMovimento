# Análise Dinâmica via Método Híbrido dos Elementos Finitos

Reimplementação em Python da dissertação de mestrado de **Rodrigo Nascimento Barros**
(PUC-Rio, 2017): *"Análise Dinâmica de Treliças e Pórticos Tridimensionais usando uma
Técnica Avançada de Superposição Modal"*.

Programa original em MAPLE CLASSIC, reescrito aqui em Python com:
- Versão básica em `numpy/scipy` (float64)
- **Versão de alta precisão** (`hp/`) em `mpmath` com 50-60 dígitos decimais
- **Versão amortecida** (`damped/`) com Rayleigh e Newmark-β
- **Análise Fuzzy-TOPSIS** classificando exemplos

---

## Estado atual da implementação

### O que está validado
| Item | Status |
|---|---|
| Método híbrido FEM (Hellinger-Reissner) | ✅ Implementado e validado |
| Viga em balanço vs solução analítica (3MM) | ✅ **0.0005% de erro** |
| K(ω) fechada (truss, beam Williams-Wittrick, torsion) | ✅ |
| Linearização companion 1MM, 2MM, 3MM, 4MM | ✅ |
| M3-M6 analíticos via `mpmath.taylor` | ✅ |
| Versão com amortecimento (Rayleigh + Newmark) | ✅ |
| Fuzzy-TOPSIS (4 alternativas) | ✅ |

### O que ainda precisa de trabalho
| Item | Onde está | O que falta |
|---|---|---|
| Ex 01 (Treliça Weaver) | `hp/examples_disc.py:ex01_*` | Geometria/restrições da Fig 5.1 não coincidem 100% com a tese (erros ~30%) |
| Ex 03 (Pórtico 2D Weaver) | `hp/examples_disc.py:ex03_4nos` | Geometria da Fig 5.19 (pórtico em V invertido) interpretada errada — erros ~350% |
| Ex 06 modos 2-3 (Treliça 3D Paz) | `hp/examples_disc.py:ex06_1no` | Pirâmide simétrica gera modos degenerados; tese tem 3 freqs distintas → geometria base é assimétrica |
| 5MM e 6MM | `hp/nmm_solver.py:solve_nmm_companion` | Matriz companion mal-condicionada com 60 dígitos. Solução: aumentar `mp.dps ≥ 120` |
| Discretizações de 24+ nós | `hp/examples_disc.py` | Performance: assembly em mpmath é O(n²) em Python puro; >50 nós inviável |

---

## Estrutura do projeto

```
python/
├── elements.py            # versão básica (float64): K(ω), K0, M1, M2 dos elementos
├── assembly.py            # Structure, montagem, BC
├── solver.py              # solvers 1MM, 2MM (float64)
├── examples.py            # exemplos da tese em float64
├── main.py                # CLI
│
├── hp/                    # ALTA PRECISÃO (mpmath, 50-60 dígitos)
│   ├── core.py            # elementos, K0, M1, M2, solvers 1MM e 2MM
│   ├── examples.py        # exemplos com parâmetros originais
│   ├── examples_disc.py   # exemplos com TODAS as discretizações da tese
│   ├── examples_refined.py# variantes subdivididas para fuzzy-TOPSIS
│   ├── nmm_solver.py      # M3-M6 via Taylor + companion polinomial
│   ├── validation.py      # validação 1MM/2MM vs tese
│   └── full_validation.py # pipeline completo 1-4MM (entry point)
│
├── damped/                # VERSÃO AMORTECIDA (paralela, não substitui)
│   ├── core.py            # Rayleigh, autovalor quadrático, Newmark-β
│   └── validation.py
│
└── fuzzy_topsis.py        # análise Fuzzy-TOPSIS
```

---

## Como rodar (Linux/Windows/macOS)

### Setup
```bash
python -m venv .venv && source .venv/bin/activate  # ou .venv\Scripts\activate
pip install numpy scipy matplotlib mpmath PyPDF2 pymupdf
```

### Reprodução das tabelas da tese (alta precisão)
```bash
# Pipeline completo: 1MM-4MM em todos os exemplos com discretizações
python -m hp.full_validation --nmm 4

# Apenas viga em balanço (validação contra analítica)
python -m hp.validation
```

### Versão amortecida
```bash
python -m damped.validation
```

### Análise Fuzzy-TOPSIS
```bash
python fuzzy_topsis.py
```

### Versão básica (float64, mais rápida)
```bash
python main.py --all                # todos os exemplos
python main.py --validate           # viga em balanço
python main.py --example 4 --nmm 2  # exemplo específico
```

---

## Próximos passos sugeridos (para próxima sessão)

### Prioridade alta
1. **Corrigir geometria do Ex 03** (Pórtico 2D Weaver)
   - Acessar Fig 5.19 da tese em `Downloads/rodrigo mestrado.PDF` (página 60).
   - Reinterpretar: tese fala de "6 barras prismáticas" com L=0.762m.
   - O pórtico parece ser portal frame simples com diagonal, NÃO V invertido.
   - Testar variantes de geometria até bater com Tabela 5.9 (modo 1 = 89.39 rad/s).

2. **Estender solver para 5MM e 6MM**
   - Em `hp/nmm_solver.py:solve_nmm_companion`, adicionar `mp.dps = 120` no início.
   - Filtrar mais agressivamente autovalores espúrios (imag muito grande).
   - Validar que viga em balanço continua dando 0.0001% de erro.

3. **Corrigir geometria do Ex 06** (Treliça 3D Paz)
   - Fig 5.34 mostra base 2.54×2.54m, altura 1.27m.
   - Para obter 3 freqs distintas (32.84, 69.15, 98.95 Hz), base deve ser assimétrica.
   - Sugestão: testar octaedro (4 base + 1 topo + simetria quebrada por massa concentrada).

### Prioridade média
4. **Acelerar montagem em mpmath**
   - Usar `mpmath.fp` (float64 com formatação mpf) onde precisão extra não é necessária.
   - Ou: assembly em numpy → conversão para mpmath só para solver final.
   - Atualmente 24 nós (90 GDLs) inviável; meta: 5 segundos para 24 nós em 4MM.

5. **Reproduzir Ex 02 (Treliça Weaver simétrica, 8 nós)**
   - Não foi implementado nesta sessão.
   - Dados: alumínio E=69 GPa, ρ=2620 kg/m³, A_vh=1.5in² e A_diag=0.5in²
     (verificar conversões), L=5m.
   - Geometria: ponte Pratt-like com 4 vãos.

### Prioridade baixa
6. **Reproduzir Ex 05 (Pórtico Petyt, 8 nós)**
   - E=219.9 GN/m², ρ=7850, L=1m, 12 barras, 8 nós livres.

7. **Resposta no tempo via Newmark-β** (já implementado em `damped/core.py`)
   - Validar contra Figs 5.2, 5.21, 5.22 da tese (deslocamento × tempo).

---

## Referências da tese

- Dissertação: `Downloads/rodrigo mestrado.PDF` (88 páginas)
- Texto extraído: `Downloads/thesis_full.txt` (use PyPDF2 ou pymupdf)
- Figuras: `Downloads/thesis_figs/` (PNG de cada página, gerados com pymupdf)

### Tabelas críticas
- **Tabela 5.1**: Treliça Weaver 1MM vs Weaver → ω = [420.51, 1168.20, 1864.34] rad/s
- **Tabela 5.3**: Treliça Weaver TQ 1MM-6MM (3 freqs)
- **Tabela 5.4**: Treliça 5 nós 1MM-6MM (13 freqs total)
- **Tabela 5.5**: Pórtico Weaver 8 nós (Ex 02, em **Hz**) → Modo 1 = 79.55 Hz
- **Tabela 5.9**: Pórtico Weaver 4 nós 1MM-6MM (12 freqs)
- **Tabela 5.12**: Pórtico 3D Paz 1MM-6MM (6 freqs) ← **reproduzido com 3% erro**
- **Tabela 5.16**: Treliça 3D Paz 1MM (3 freqs em Hz) ← **modo 1: 0.67% erro**
- **Tabela 5.18/5.19**: Treliça 3D Paz TQ/TC com 1MM-6MM

### Livros base (em `Downloads/`)
- `148855732-Matrix-Analysis-OfbFramed-Structures.pdf` — Weaver & Gere (1980)
- `135434492-Franklin-Y-Cheng-Matrix-Analysis-of-Structural-BookFi-org.pdf` — Cheng

---

## Notas técnicas

### Conversões críticas (sistema técnico → SI)
A tese usa Paz [15] como referência, que usa unidades técnicas antigas:
- **m em kg·s²/m²** (Tabela 5.11 - Ex 04): multiplicar por g=9.80665 para obter ρ_SI = m·g/A
- **m em kg·s²/m** (Tabela 5.16 - Ex 06): m_si_per_length = m × g
- Sem essa conversão, frequências saem √g ≈ 3.13× maiores que a tese.

### Solver Companion Polinomial
Para (K0 - λM1 - λ²M2 - ... - λⁿMₙ)·Φ = 0 com λ = ω², monta:
```
A·z = λ·B·z
A = [[0     I     ... 0   ],
     [...               ],
     [K0   -M1   ... -M_{n-1}]]
B = block_diag(I, I, ..., I, M_n)
```
Resolver via `mp.eig(B⁻¹·A)`. Funciona estável até 4MM em 60 dígitos.

### Performance esperada (ambiente: Windows 11, Python 3.13, mpmath 1.3, 8 GB RAM)
- Viga em balanço 4MM (10 elem, 30 GDL): ~23 s
- Ex 04 Paz 4MM (1 nó, 6 GDL): ~3.5 s
- Ex 06 Paz 4MM (1 nó, 3 GDL): ~0.5 s
- Pipeline `full_validation`: ~135 s

---

## Filosofia do código

- **Não substituir**: a versão básica `elements.py` continua funcional; `hp/` e `damped/` são paralelas.
- **Determinismo**: tudo é reproduzível com `mp.dps = 50` ou superior.
- **Sem dependências obscuras**: numpy, scipy, mpmath, matplotlib.
- **Documentar em português**: textos explicativos e comentários em PT-BR seguindo a tese.

---

## Continuidade do trabalho

Esta sessão entregou o pipeline 1MM-4MM funcional, a infraestrutura para 5MM-6MM
(precisa aumentar `mp.dps`) e o Fuzzy-TOPSIS. As próximas sessões devem focar em:

1. Refinar geometrias dos Ex 01, 03 e 06 acessando as figuras do PDF (`fitz`/pymupdf).
2. Estender para 5MM/6MM com `mp.dps = 120`.
3. Validar contra TODAS as tabelas (5.1 a 5.21) com erro < 1% nos modos principais.
4. Implementar resposta no tempo com Newmark-β e comparar com Figs 5.2-5.22.

Quando rodar `full_validation.py`, deve produzir as colunas `Δ% tese` mostrando
convergência. Se aparecerem erros > 5% nos modos 1-3 do Ex 04 ou Ex 06, há
regressão.
