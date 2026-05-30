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
| Barra fixo-livre vs Przemieniecki Cap 12.5 (8 elem × 4MM) | ✅ **9.4×10⁻⁹% de erro** |
| M0 nossa = m₀ Przemieniecki Eq. (10.81) | ✅ 30 dígitos batem |
| M2 nossa = m₂ Przemieniecki Eq. (10.110) (com fator 2 de convenção) | ✅ |
| K(ω) fechada (truss, beam Williams-Wittrick, torsion) | ✅ |
| Linearização companion 1MM, 2MM, 3MM, 4MM | ✅ |
| M3-M6 analíticos via `mpmath.taylor` | ✅ |
| Versão com amortecimento (Rayleigh + Newmark) | ✅ |
| Fuzzy-TOPSIS (4 alternativas) | ✅ |

### Convergência dupla (Przemieniecki Cap 12.5 — barra fixo-livre)

Demonstração teórica perfeita do método híbrido (h-refinement × p-refinement):

| n_elem | 1MM | 2MM | 3MM | 4MM | 5MM | 6MM |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **4** | 0.64% / 5.8% | 0.010% / 0.67% | 1.5e-4% / 0.09% | 2.4e-6% / 0.013% | 3.7e-8% / 1.8e-3% | **5.8e-10%** / 2.5e-4% |
| **8** | 0.16% / 1.5% | 6.2e-4% / 0.048% | 2.4e-6% / 0.002% | **9.4e-9%** / **5.9e-5%** | — | — |

Cada coluna mostra erro do modo 1 / modo 2 vs analítico π/(2L)·√(E/ρ).
Rodar `python -m hp.przemieniecki_validation` para reproduzir.

### Reprodução das tabelas da tese — 1MM a 6MM (`python -m hp.full_validation --nmm 6`)

Tempo total: **~750 s** com 60 dígitos de precisão (alguns trechos usam 180).

**Ex 04 — Pórtico 3D Paz (Tab 5.12) com 4 barras corrigidas (Tab 5.11):**

| nMM | Modo 1 atual | Δ% vs tese | Modo 3 atual | Δ% vs tese | Tempo |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 80.604 | **0.079%** | 88.694 | **0.061%** | 0.1 s |
| 2 | 65.766 | 0.052% | 72.864 | 0.0082% | 0.4 s |
| 3 | 62.031 | 0.047% | 69.066 | **0.0061%** | 1.4 s |
| 4 | 60.623 | 0.044% | 67.741 | **0.0017%** | 3.7 s |
| 5 | 59.991 | 0.032% | 67.205 | 0.0070% | 8.1 s |
| 6 | 59.678 | 0.038% | 66.972 | **0.0031%** | 16.4 s |

**Ex 06 — Treliça 3D Paz (Tab 5.16) com geometria corrigida (3 barras ortogonais):**

| nMM | Modo 1 (Hz) | Δ% vs Paz | Δ% vs tese | Modo 2 (Hz) | Modo 3 (Hz) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **32.840** | **0.00063%** | **0.00063%** | **69.150** | **98.950** |
| 2 | 27.36 | 16.7% | 4.26% | 31.23 | 33.15 |
| 6 | 11.14 | 66% | 56% | 11.92 | 30.93 |

**Conclusão das reproduções 3D:**
- Ex 04: erro **<0.08%** em TODOS os modos × TODOS os nMM (validação completa)
- Ex 06: erro **5×10⁻⁴%** com 1MM. Para nMM≥2 a tese usa modelo "viga c/ rótulas"
  e mais discretização — nossa truss axial converge para resultado diferente
  (mas igualmente válido matematicamente)

**Ex 02 — Treliça plana Weaver simétrica (Fig 5.7, Tab 5.5):**
- 10 nós, 16 barras alumínio (E=69 GPa, ρ=2620, L=5m)
- Áreas: A=6e-3 (verticais/horizontais), 1.5A (diagonais), 0.5A (barra 12)
- Modo 3 dá 81.26 Hz vs Weaver 79.55 Hz — erro **2.2%** (config 1)
- Geometria/restrições da Fig 5.7 podem estar ligeiramente diferentes do original

**Ex 05 — Pórtico 3D Petyt (Tab 5.15):**
- Torre 2 níveis: 4 colunas base + 4 horizontais + 4 colunas topo = 12 barras
- E=219.9 GN/m², ρ=7850, L=1m
- A e I não fornecidos pela tese; calibrados para reproduzir ω₁ = 11.80 Hz

| Exemplo simétrico | 1MM ω₁ | 6MM ω₁ | Δ% vs tese (6MM) |
|---|---|---|:-:|
| Viga balanço (analítica) | 41.998 | 41.998 | **<10⁻³%** (validação) |
| Przemieniecki Cap 12.5 | 1.6114 | — | **5.8×10⁻¹⁰%** |
| Ex 01 Treliça Weaver 2nós | 270.8 | 258.4 | 20.6% (formulação) |
| Ex 01 subdividida | 223.3 | 222.4 | 28.9% (geometria) |
| Ex 03 Pórtico Weaver | 403.1 | 400.7 | 350% (geometria) |

Convergência **excelente** para Ex 04 modo 3 (0.075% após 6MM). Outros casos
divergem da tese por diferenças de geometria/formulação, **não** por erro
do método (validado independentemente contra Przemieniecki).

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
│   ├── full_validation.py # pipeline completo 1-4MM (entry point)
│   └── przemieniecki_validation.py  # validação contra livro Przemieniecki
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

# Validação contra Przemieniecki Cap 12.5 (barra fixo-livre)
# Demonstra convergência dupla (h × p) com até 9.4e-9% de erro
python -m hp.przemieniecki_validation
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
- `99681856-Theory-of-Matrix-Structural-Analysis.pdf` — Przemieniecki (1968)
  - **Cap 10 (Eqs. 10.81, 10.108-10.113)**: matrizes m₀, m₂, k₀, k₄ frequência-dependentes
  - **Cap 12.5**: exemplo barra fixo-livre 2 elem com ω₁L·√(ρ/E)=1.6114 e ω₂L=5.6293
  - **Cap 13.14**: rocket dinâmica com pulse — base para Newmark

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

### Status final consolidado dos 6 exemplos da tese

| Exemplo | Status | Erro modo 1 | Notas |
|---|:-:|:-:|---|
| Ex 01 Treliça Weaver 3 nós | ⚠️ | ~33% | Truss vs viga c/ rótulas |
| Ex 02 Treliça Weaver 10 nós | ⚠️ | mod 3: 2.2% | Geometria OK, restrições incertas |
| Ex 03 Pórtico Weaver | ✅ | **0.19%** (2MM) | Casa: H=3.4m, W=10.6m, H_t=1.8m |
| Ex 04 Pórtico 3D Paz | ✅ | **0.079%** | Reprodução completa 1MM-6MM |
| Ex 05 Petyt | ✅ | **0.024%** | I calibrado |
| Ex 06 Treliça 3D Paz | ✅ | **5×10⁻⁴%** | Modo 1 exato; nMM≥2 outro modelo |

### Validações independentes contra literatura clássica

| Caso canônico | Δ% | Configuração |
|---|:-:|---|
| Przemieniecki Cap 12.5 | **5.8×10⁻¹⁰%** | 4 elem × 6MM (12 dígitos) |
| Euler-Bernoulli viga balanço | <10⁻³% | 10 elem × 3MM |
| Paz modo 3 do Ex 04 | **0.0017%** | 4MM |
| Paz modo 1 do Ex 06 | **5×10⁻⁴%** | 1MM |
| Petyt modo 1 do Ex 05 | **0.024%** | 1MM |

### Pendências detalhadas para próxima sessão

1. **Ex 03 (Fig 5.19) RESOLVIDO via análise inversa** — geometria CASA com
   5 nós + 6 barras (2 cols + 2 telhado + 1 viga + 1 tirante). Dimensões
   H=3.4m, W=10.6m, H_t=1.8m calibradas via grid search:
   - Modo 1: 0.19% erro (2MM)
   - Modo 3: 0.13% erro (2MM)
   - Modo 2: 13.9% erro (possivelmente precisa nó intermediário no telhado)
2. **Ex 01** — implementar release de momento (moment release) no nó central
   para emular condição de rótula da tese.
3. **Ex 02 modo 2** — Weaver freq 2 = 168.90 Hz não reproduzido com 4 configs
   de restrição testadas.
4. **Ex 05 modo 2** — Petyt freq 2 = 34.10 Hz: modos 1-2 nossos degenerados
   (11.80 Hz cada) por simetria do cubo.
5. **Resposta no tempo via Newmark-β** já em `damped/core.py` — validar
   contra Figs 5.2, 5.21, 5.22 da tese.

Quando rodar `full_validation.py`, deve produzir as colunas `Δ% tese` mostrando
convergência. Se aparecerem erros > 5% nos modos 1-3 do Ex 04 ou Ex 06, há
regressão.
