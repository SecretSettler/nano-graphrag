import os
import logging
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import HNSWVectorStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from vllm import LLM, SamplingParams
from typing import List
import asyncio
import json
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


DATASET_DIR = "/home/jysc/ragchecker_benchmark/kiwi"
WORKING_DIR = f"./nano_graphrag_vllm_and_local_embedding_{DATASET_DIR.split('/')[-1]}"
# MAX_CONTEXT_TOKENS = 10000
# MAX_OUTPUT_TOKENS = 1000
MAX_CONTEXT_TOKENS = 10000
MAX_OUTPUT_TOKENS = 100

print("Dataset dir:", DATASET_DIR, "Working dir:", WORKING_DIR)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=MAX_CONTEXT_TOKENS + MAX_OUTPUT_TOKENS,
    device="cuda:0",
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)


def print_outputs(outputs):
    print("=" * 80)
    print("Generated reponse:")
    print(outputs)
    print("-" * 80)


async def vllm_llama_model(
    prompt: str, system_prompt: str = None, history_messages: List[dict] = [], **kwargs
) -> str:
    # openai_async_client = AsyncOpenAI(
    #     api_key=DEEPSEEK_API_KEY, base_url=API_BASE
    # )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # response = await openai_async_client.chat.completions.create(
    #     model=MODEL, messages=messages, **kwargs
    # )
    response = llm.chat(messages=messages, sampling_params=sampling_params)
    return response[0].outputs[0].text


async def vllm_llama_model_batch_inference(
    prompt: List[str],
    system_prompt: List[str] = None,
    history_messages: List[List[dict]] = [],
    **kwargs,
) -> str:
    batch_messages = []
    for i in range(len(prompt)):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt[i]})

        if history_messages:
            messages.extend(history_messages[i])
        messages.append({"role": "user", "content": prompt[i]})
        batch_messages.append(messages)
    batch_response = llm.chat(messages=batch_messages, sampling_params=sampling_params)
    return [response.outputs[0].text for response in batch_response]


EMBEDDING_MODEL = LLM(
    "intfloat/e5-mistral-7b-instruct", enforce_eager=True, device="cuda:1"
)

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL.llm_engine.model_config.get_hidden_size(),
    max_token_size=EMBEDDING_MODEL.llm_engine.model_config.max_model_len,
)
async def local_embedding(texts: list[str]) -> list[list[float]]:
    outputs = EMBEDDING_MODEL.encode(texts)
    return [output.outputs.embedding for output in outputs]

# EMBEDDING_MODEL = SentenceTransformer(
#     'Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True, cache_folder=WORKING_DIR, device="cuda:1"
# )


# # We're using Sentence Transformers to generate embeddings for the BGE model
# @wrap_embedding_func_with_attrs(
#     embedding_dim=EMBEDDING_MODEL.get_sentence_embedding_dimension(),
#     max_token_size=EMBEDDING_MODEL.max_seq_length,
# )
# async def local_embedding(texts: list[str]) -> np.ndarray:
#     return EMBEDDING_MODEL.encode(texts, normalize_embeddings=True)


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def insert():
    from time import time

    # graph = pickle.load(open(os.path.join(DATASET_DIR, "graph.pkl"), "rb"))
    corpus = []
    with open(os.path.join(DATASET_DIR, 'corpus.jsonl'), 'r') as file:
        for line in file:
            corpus.append(json.loads(line)['text'])

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=vllm_llama_model,
        cheap_model_func=vllm_llama_model,
        embedding_func=local_embedding,
        best_model_max_token_size=MAX_CONTEXT_TOKENS,
        cheap_model_max_token_size=MAX_CONTEXT_TOKENS,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={
            "max_elements": 10000000,
            "ef_search": 200,
            "M": 50,
        },
    )
    start = time()
    rag.insert(corpus)
    print("indexing time:", time() - start, "s")

def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=False,
        best_model_func=vllm_llama_model_batch_inference,
        cheap_model_func=vllm_llama_model_batch_inference,
        embedding_func=local_embedding,
        best_model_max_token_size=MAX_CONTEXT_TOKENS,
        cheap_model_max_token_size=MAX_CONTEXT_TOKENS,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={
            "max_elements": 10000000,
            "ef_search": 200,
            "M": 50,
        },
    )
    # print the graph
    num_nodes = asyncio.run(rag.chunk_entity_relation_graph.nodes())
    num_edges = asyncio.run(rag.chunk_entity_relation_graph.edges())
    print(f"Graph # of nodes: {len(num_nodes)}, # of edges: {len(num_edges)}")
    exit()
    batch_response = rag.batch_query(
        ["What are the prior approaches proposed to improve faithfulness of the reasoning steps generated by LLMs and what tasks are they applied on?" for _ in range(1)],
        param=QueryParam(
            mode="local",
            top_k=20,
            edge_depth=3,
            # local_max_token_for_text_unit=int(MAX_CONTEXT_TOKENS * 0.3),
            local_max_token_for_local_context=int(MAX_CONTEXT_TOKENS * 0.3),
            # local_max_token_for_community_report=int(MAX_CONTEXT_TOKENS * 0.3),
            response_type="Single phrase or a sentence, concise and no redundant explanation needed.",
        ),
    )
    for reponse in batch_response:
        print_outputs(reponse)


if __name__ == "__main__":
    insert()
    # query()
