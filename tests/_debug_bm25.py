from harness.registry import ToolSpec, schema_for_fields
from harness.retrieval import BM25Retriever, retrieval_query, tokenize
from rank_bm25 import BM25Okapi


def make(name: str, phrase: str) -> ToolSpec:
    app, action = name.split(".")
    return ToolSpec(
        name=name,
        app=app,
        action=action,
        phrase=phrase,
        description=phrase,
        side_effect=False,
        parameters=schema_for_fields(["title", "note"]),
        retrieval_text=f"{name.replace('.', ' ')} {phrase}",
    )


tools = [
    make("calendar.create", "create the event in calendar"),
    make("notes.create", "create the note in notes"),
    make("calendar.list", "list the event in calendar"),
]
query = "create the event in calendar title='Case 1' note='alpha detail'"
print("Q", retrieval_query(query))
print("tokens", tokenize(retrieval_query(query)))
docs = [tokenize(tool.retrieval_text) for tool in tools]
print("docs", docs)
scores = BM25Okapi(docs).get_scores(tokenize(retrieval_query(query)))
print("scores", list(scores))
print("ranked", BM25Retriever(tools).retrieve(query, k=3))
