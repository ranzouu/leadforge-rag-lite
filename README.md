# LeadForge RAG Lite

Pipeline de **RAG local** para consultar políticas de prospecção por nicho. Usa corpus local, busca por similaridade e geração com LLM.

## Pergunta mensurável
Quão bem o retrieval recupera a política correta para uma pergunta de negócio?

## O que ele prova
- Entendimento de **RAG**: ingestão, chunking, retrieval, geração
- Métricas de retrieval: **MRR@3** e **hit@3**
- Reprodutibilidade: script único gera CSV de resultados

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Executar
```bash
# Demo offline (sem LLM)
python -m leadforge_rag_lite --demo

# Com LLM
export LLM_API_KEY="..."
export LLM_BASE_URL="https://api.openai.com/v1"
python -m leadforge_rag_lite
```

## Output
- `runs/latest.csv` — resultados por query
- `runs/summary.csv` — MRR@3 e hit@3
