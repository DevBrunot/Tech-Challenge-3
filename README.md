# Tech Challenge Fase 3 - Machine Learning Engineering

Projeto em Python Tech Challenge:

- **EDA:** estrutura, nulos, estatísticas e gráficos atraso por origem, companhia, hora.
- **Engenharia de features:** `PERIODO_DIA`, `DIA_SEMANA`, one-hot ORIGIN, DEST, CARRIER, PERIODO_DIA.
- **Regressão:** prever **quanto tempo o atraso vai durar** min com **Ridge** e **Random Forest**; métricas MAE, RMSE, R²; comparação de dois algoritmos.
- **Modelagem não supervisionada:** clusterização de **voos** perfil atraso partida/chegada e de **aeroportos** volume e atraso; gráficos e interpretação.
- **Respostas de negócio:** aeroportos e companhias mais críticos; dia da semana; perguntas guia do desafio.
- **Apresentação crítica:** conclusões, limitações e próximos passos em `CONCLUSOES_E_LIMITACOES.md`.

## Estrutura

- `projeto_atrasos_voos.py` -> script principal
- `requirements.txt` -> dependencias
- `output/figuras` -> graficos gerados
- `output/tabelas` -> tabelas e metricas em CSV

## Como executar no Windows (PowerShell)

1) Entre na pasta do projeto:

```powershell
cd "C:\Users\bruno.cordeiro\PROJETOS BRUNO\pos 3 - TECH CHALLENGE"
```

2) Crie o ambiente virtual:

```powershell
python -m venv .venv
```

3) Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

4) Instale as dependencias:

```powershell
pip install -r requirements.txt
```

5) Rode o projeto:

```powershell
python projeto_atrasos_voos.py
```

## Dataset

O script tenta carregar:

1. `flights.csv` local (na mesma pasta), ou
2. baixar de uma URL publica automaticamente.

Se quiser usar um CSV proprio, basta salvar com o nome `flights.csv` ao lado do script.

## Entregáveis gerados

- **Figuras:** `output/figuras` (EDA, real vs predito da regressão, clusters de voos e de aeroportos).
- **Tabelas:** `output/tabelas` (valores ausentes, atraso por origem/companhia/hora, comparação dos modelos de regressão, resumos de clusters, respostas de negócio).
- **Apresentação crítica:** `CONCLUSOES_E_LIMITACOES.md` (conclusões, limitações, melhorias e próximos passos).

**Entregáveis do desafio:** repositório com código completo + vídeo de apresentação (5–10 min) explicando o trabalho, resultados e conclusões.
