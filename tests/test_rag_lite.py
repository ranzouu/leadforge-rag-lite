from leadforge_rag_lite.experiment import _load_policies, _load_queries, retrieve


def test_load_policies():
    policies = _load_policies()
    assert len(policies) >= 1


def test_load_queries():
    queries = _load_queries()
    assert len(queries) >= 1


def test_retrieval_hit():
    policies = _load_policies()
    queries = _load_queries()
    top3 = retrieve(policies, queries[0].query, top_k=3)
    assert len(top3) <= 3
    assert top3[0][1] >= 0
