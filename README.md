# LeadForge RAG Lite

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit-learn-1.5%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)

Pipeline de RAG leve para consultar políticas de prospecção por nicho. Retrieval com TF-IDF, métricas `hit@3` e `MRR@3`, e modo demo offline.

## Sobre

O `leadforge-rag-lite` demonstra um fluxo completo de **Retrieval-Augmented Generation (RAG)** em formato mínimo, reprodutível e medível:

- **Ingestão**: corpus local de políticas por nicho (`data/policies.json`)
- **Retrieval**: ranqueamento por similaridade TF-IDF (top-3)
- **Geração**: resposta objetiva em português, com suporte a LLM ou modo demo offline
- **Avaliação**: métricas de retrieval (`hit@3` e `MRR@3`) exportadas em CSV

Ideal para portfólio, pois separa claramente lógica de retrieval, geração e experimentação.

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Linguagem | Python 3.11+ |
| Retrieval | TF-IDF customizado (`experiment.py`) |
| Schemas | Pydantic v2 |
| Cliente HTTP | httpx |
| Métricas / CSV | pandas |
| CLI / Output | rich |
| Testes | pytest |

## Features

- **RAG completo em um único módulo** — sem bancos vetoriais externos
- **Modo demo offline** — gera respostas sem precisar de API key
- **Modo LLM** — plugável via `LLM_API_KEY` + `LLM_BASE_URL`
- **Métricas automáticas** — `hit@3` e `MRR@3` calculadas por query
- **Reprodutibilidade** — experimentos salvos em `runs/latest.csv` e `runs/summary.csv`
- **Testes de retrieval** — validação de carregamento e hit rate

## Arquitetura

```
Pergunta do usuário
        │
        ▼
┌─────────────┐
│   Retrieval │  TF-IDF sobre policies.json → top-3 políticas
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Geração    │  Prompt com contexto + resposta objetiva em PT-BR
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Métricas   │  hit@3, MRR@3, latência → CSV
└─────────────┘
```

## Exemplos

### Demo offline (sem LLM)

```bash
python -m leadforge_rag_lite --demo
```

### Com LLM real

```bash
export LLM_API_KEY="sua-chave"
export LLM_BASE_URL="https://api.openai.com/v1"
python -m leadforge_rag_lite
```

## Métricas

A avaliação usa queries de exemplo sobre nichos: **academia**, **pet** e **restaurante**.

| queries | hit@3 | MRR@3 | avg latency |
|---------|-------|-------|-------------|
| 3 | 3 | 0.8333 | 0.07ms |

- **hit@3**: a política esperada aparece entre os top-3 resultados?
- **MRR@3**: média do recíproco da posição do primeiro acerto (1.0 = sempre na 1ª posição)

## Como rodar

```bash
git clone https://github.com/ranzouu/leadforge-rag-lite.git
cd leadforge-rag-lite
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m leadforge_rag_lite --demo
```

## Testes

```bash
pytest
```

## Estrutura

```
leadforge-rag-lite/
├── data/
│   ├── policies.json       # Corpus de políticas por nicho
│   └── eval_queries.json   # Queries de avaliação com esperado
├── src/
│   └── leadforge_rag_lite/
│       ├── __main__.py
│       ├── cli.py
│       └── experiment.py   # Retrieval, LLM, métricas e CSV
├── tests/
│   └── test_rag_lite.py
├── runs/
│   ├── latest.csv          # Resultados por query
│   └── summary.csv         # Resumo agregado
├── pyproject.toml
└── README.md
```

## Licença

MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.
