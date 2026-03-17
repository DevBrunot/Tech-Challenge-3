# Tech Challenge - Projeto em venv (sem Jupyter)

Este projeto executa todo o pipeline de Ciencia de Dados em um script Python:

- EDA (estrutura, nulos, estatisticas e graficos)
- Engenharia de features (`PERIODO_DIA`, `DIA_SEMANA`, one-hot)
- Modelagem supervisionada (`LogisticRegression` e `RandomForest`)
- Modelagem nao supervisionada (`KMeans`)
- Exportacao de resultados e figuras

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

## Entregaveis gerados

Ao final da execucao, voce tera:

- Graficos em `output/figuras`
- Tabelas e metricas em `output/tabelas`

Isso facilita sua apresentacao sem depender de notebook.
