#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from dotenv import load_dotenv
import numpy as np
from numpy.linalg import norm
from google import genai
import pennylane as qml
from pennylane import numpy as pnp

load_dotenv()

# ==========================================
# 1. Gemini Client
# ==========================================
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ==========================================
# 2. Your Documents
# ==========================================
documents = [
    "Our company refund policy allows customers to return products within 30 days of purchase for a full refund.",
    "Refunds are processed within 5-7 business days after we receive the returned item.",
    "Shipping usually takes 5-7 business days across India. Express shipping is available for 2-day delivery.",
    "International shipping is available to select countries, taking 10-15 business days.",
    "We offer 24/7 customer support through email and chat. Phone support is available from 9 AM to 6 PM.",
    "Our support team can be reached via WhatsApp for urgent order issues.",
    "All our products come with a 1-year manufacturer warranty covering technical defects.",
    "Extended warranty plans up to 3 years are available for an additional fee.",
    "You can track your order using the tracking link sent to your email after dispatch.",
    "If your tracking link isn't working, contact support with your order ID.",
    "We accept payments via credit card, debit card, UPI, and net banking.",
    "EMI options are available on orders above ₹5000.",
    "Gift wrapping is available at checkout for an additional ₹49.",
    "We do not offer refunds on sale or clearance items unless the product is defective.",
    "Bulk orders above 50 units qualify for a 10% discount — contact our sales team.",
]

# ==========================================
# 3. Classical Embedding + Similarity
# ==========================================
def get_embeddings_batch(texts_list, batch_size=100):
    all_embeddings = []

    # Chunk text list into batches of 100 to stay safely within payload limits
    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i : i + batch_size]
        result = client.models.embed_content(
            model="gemini-embedding-001", contents=batch
        )
        # Extract the vector list from each returned embedding
        all_embeddings.extend([e.values for e in result.embeddings])

    return all_embeddings

def get_embedding(text):
    """Single-text convenience wrapper around get_embeddings_batch."""
    return get_embeddings_batch([text])[0]

import time
from google.genai.errors import ClientError

_query_embedding_cache = {}

def get_embedding_cached(text):
    if text not in _query_embedding_cache:
        for attempt in range(3):
            try:
                _query_embedding_cache[text] = get_embedding(text)
                break
            except ClientError as e:
                if e.status_code == 429 and attempt < 2:
                    print("Rate limited, waiting 30s...")
                    time.sleep(30)
                else:
                    raise
    return _query_embedding_cache[text]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

print("Creating embeddings for documents...")
doc_embeddings = get_embeddings_batch(documents)
print("Embeddings created!\n")

# NOTE: RETRIEVE_TOP_K controls how many candidates the classical retriever
# hands to the quantum reranker. The reranker's top_k is set to match this
# value everywhere it's called, so the quantum stage only ever REORDERS
# these candidates -- it never drops one. This removes the "correct answer
# silently disappears" failure mode entirely, rather than patching around
# it with a safety net.
RETRIEVE_TOP_K = 3

def retrieve(query, top_k=RETRIEVE_TOP_K):
    query_embedding = get_embedding_cached(query)
    scores = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[:top_k]

# ==========================================
# 4. Quantum Kernel Part
# ==========================================
n_qubits = 8

# ----- Choose Device -----
# Option A: Fast local simulator (recommended while developing)
dev = qml.device("default.qubit", wires=n_qubits)

# Option B: Real IBM Quantum Hardware (uncomment to use)
# NOTE: IBM's Open Plan does not permit Session-based execution
# (see qiskit_ibm_runtime.Session) -- only job/batch mode is allowed.
# qml.device("qiskit.remote", ...) creates a Session internally and will
# fail with a 400 error ("not authorized to run a session when using the
# open plan") on the Open Plan. Running on real hardware currently
# requires bypassing this PennyLane device and calling SamplerV2 in job
# mode directly against a manually-built Qiskit circuit.
#
# from qiskit_ibm_runtime import QiskitRuntimeService
# service = QiskitRuntimeService()          # uses your saved IBM account
# backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
# dev = qml.device("qiskit.remote", wires=n_qubits, backend=backend)
# print(f"Using real IBM backend: {backend.name}")

def feature_map(x):
    x = x[:n_qubits] if len(x) >= n_qubits else np.pad(x, (0, n_qubits - len(x)))
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(n_qubits))

def quantum_kernel(x1, x2):
    probs = kernel_circuit(x1, x2)
    return float(probs[0])

def reduce_embedding(embedding, target_dim=n_qubits):
    """
    Compress a high-dimensional embedding down to `target_dim` values by
    chunk-averaging, then rescale to span roughly [-pi, pi].

    The rescaling step matters: gemini-embedding-001 components are small
    (~0.01-0.05 in magnitude), which without rescaling produces near-zero
    rotation angles -- RY/RZ gates that are nearly identity regardless of
    input, causing the kernel to return ~1.0 for every pair of documents
    (no discrimination at all). Chunk-averaging (vs. "first few dims + one
    big mean of the rest") also spreads signal more evenly across all
    n_qubits instead of wasting several qubits on a single averaged blob.
    """
    embedding = np.array(embedding)
    if len(embedding) <= target_dim:
        reduced = embedding
    else:
        chunks = np.array_split(embedding, target_dim)
        reduced = np.array([chunk.mean() for chunk in chunks])

    reduced = reduced - reduced.mean()
    max_abs = np.max(np.abs(reduced)) or 1e-8
    reduced = (reduced / max_abs) * np.pi
    return reduced

doc_to_embedding = dict(zip(documents, doc_embeddings))

def quantum_rerank(query_embedding, candidate_docs, classical_scores, top_k=RETRIEVE_TOP_K, alpha=0.85):
    """
    Rerank candidates by a blend of classical cosine similarity and the
    quantum kernel score: combined = alpha * classical + (1 - alpha) * quantum.

    top_k defaults to RETRIEVE_TOP_K (the full candidate set) so this
    function REORDERS candidates rather than filtering any out. Passing a
    smaller top_k turns this into a filter again, which was shown (via
    alpha_sweep) to drop the correct document from context in 2/3 test
    queries at alpha <= 0.6 when only 2 of 3 candidates were kept.
    """
    query_red = reduce_embedding(query_embedding)
    scores = []
    for doc, c_score in zip(candidate_docs, classical_scores):
        doc_emb = doc_to_embedding[doc] 
        doc_red = reduce_embedding(doc_emb)
        q_score = quantum_kernel(query_red, doc_red)
        combined = alpha * c_score + (1 - alpha) * q_score
        scores.append((combined, q_score, c_score, doc))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_k]


def sanity_check_kernel():
    test_doc = documents[0]
    test_emb = get_embedding(test_doc)
    test_red = reduce_embedding(test_emb)
    self_score = quantum_kernel(test_red, test_red)
    print(f"\n[Sanity check] Self-similarity score (should be ~1.0): {round(self_score, 4)}")

sanity_check_kernel()


# ==========================================
# 5. Hybrid RAG Function
# ==========================================
def ask_rag(question, alpha=0.85):
    print(f"Question: {question}\n")

    # Step 1: Classical retrieval
    classical_results = retrieve(question, top_k=RETRIEVE_TOP_K)
    candidate_docs = [doc for score, doc in classical_results]
    classical_scores = [score for score, doc in classical_results]

    print("Classical Top Documents:")
    for score, doc in classical_results:
        print(f"- ({round(score, 3)}) {doc}")

    # Step 2: Quantum Re-ranking (reorder-only -- top_k matches candidate count)
    query_emb = get_embedding_cached(question) 
    quantum_results = quantum_rerank(
        query_emb, candidate_docs, classical_scores,
        top_k=RETRIEVE_TOP_K, alpha=alpha
    )

    print("\nQuantum Re-ranked Documents:")
    for combined, q_score, c_score, doc in quantum_results:
        print(f"- (combined: {round(combined, 3)}, quantum: {round(q_score, 3)}, classical: {round(c_score, 3)}) {doc}")

    # No safety net needed: top_k == len(candidate_docs), so every classical
    # candidate is guaranteed to appear in context, just possibly reordered.
    context = "\n".join(doc for _, _, _, doc in quantum_results)

    # Step 3: Generate answer
    prompt = f"""
Answer the question based only on the following context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    print("\nFinal Answer:")
    print(response.text)


# ==========================================
# 6. Alpha Sweep Harness (run manually when needed)
# ==========================================
def alpha_sweep(queries, alphas=(0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0), top_k=RETRIEVE_TOP_K):
    """
    NOTE: with top_k == RETRIEVE_TOP_K (reorder-only), "classical_top1_survived"
    is trivially always True -- nothing can be dropped. This harness is kept
    for cases where you want to test a genuine filtering top_k < RETRIEVE_TOP_K
    (e.g. against a larger candidate pool). For the reorder-only default,
    track "combined_top1_matches_classical" instead to see how much the
    quantum term shifts the #1 position.
    """
    results = []
    for question in queries:
        classical_results = retrieve(question, top_k=RETRIEVE_TOP_K)
        candidate_docs = [doc for score, doc in classical_results]
        classical_scores = [score for score, doc in classical_results]
        classical_top1_doc = classical_results[0][1]

        query_emb = get_embedding_cached(question) 

        for alpha in alphas:
            reranked = quantum_rerank(query_emb, candidate_docs, classical_scores, top_k=top_k, alpha=alpha)
            reranked_docs = [doc for _, _, _, doc in reranked]
            survived = classical_top1_doc in reranked_docs
            top1_matches = reranked_docs[0] == classical_top1_doc

            results.append({
                "question": question,
                "alpha": alpha,
                "classical_top1_survived": survived,
                "combined_top1_matches_classical": top1_matches,
            })
    return results


ground_truth_queries = [
    {
        "question": "I want to add gift wrapping to my order — is there an extra fee?",
        "correct_doc": "Gift wrapping is available at checkout for an additional ₹49.",
    },
    {
        "question": "Does extending my coverage cost extra?",
        "correct_doc": "Extended warranty plans up to 3 years are available for an additional fee.",
    },
    {
        "question": "Is there a charge for splitting my payment into installments?",
        "correct_doc": "EMI options are available on orders above ₹5000.",
    },
]

def ground_truth_sweep(cases, alphas=(0.0, 0.2, 0.5, 0.8, 1.0)):
    results = []
    for case in cases:
        question, correct_doc = case["question"], case["correct_doc"]
        classical_results = retrieve(question, top_k=RETRIEVE_TOP_K)
        candidate_docs = [doc for score, doc in classical_results]
        classical_scores = [score for score, doc in classical_results]
        classical_top1_correct = classical_results[0][1] == correct_doc

        query_emb = get_embedding_cached(question)
        for alpha in alphas:
            reranked = quantum_rerank(query_emb, candidate_docs, classical_scores, top_k=RETRIEVE_TOP_K, alpha=alpha)
            reranked_top1_correct = reranked[0][3] == correct_doc
            results.append({
                "question": question, "alpha": alpha,
                "classical_top1_correct": classical_top1_correct,
                "reranked_top1_correct": reranked_top1_correct,
            })
    return results


def print_sweep_summary(results):
    alphas = sorted(set(r["alpha"] for r in results))
    questions = list(dict.fromkeys(r["question"] for r in results))

    print(f"{'Query':<45} " + " ".join(f"a={a:<4}" for a in alphas))
    for q in questions:
        row = f"{q[:43]:<45} "
        for a in alphas:
            match = next(r for r in results if r["question"] == q and r["alpha"] == a)
            row += ("  ✓   " if match["combined_top1_matches_classical"] else "  ✗   ")
        print(row)

    print("\nCombined top-1 == classical top-1, by alpha:")
    for a in alphas:
        subset = [r for r in results if r["alpha"] == a]
        rate = sum(r["combined_top1_matches_classical"] for r in subset) / len(subset)
        print(f"  alpha={a}: {rate:.0%} ({sum(r['combined_top1_matches_classical'] for r in subset)}/{len(subset)} queries)")


# ==========================================
# 7. Test
# ==========================================
ask_rag("What is your refund policy?")
ask_rag("Do you offer any discounts for bulk orders?")
ask_rag("How can I contact customer support?")

# Uncomment to run the alpha sweep: 
test_queries = [
    "What is your refund policy?",
    "Do you offer any discounts for bulk orders?",
    "How can I contact customer support?",
]
sweep_results = alpha_sweep(test_queries)
print_sweep_summary(sweep_results)

# ==========================================
# 8. Ground-truth sweep (lexical-confusion test cases)
# ==========================================
def print_ground_truth_summary(results):
    alphas = sorted(set(r["alpha"] for r in results))
    questions = list(dict.fromkeys(r["question"] for r in results))

    print(f"{'Query':<55} " + " ".join(f"a={a:<4}" for a in alphas))
    for q in questions:
        row = f"{q[:53]:<55} "
        for a in alphas:
            match = next(r for r in results if r["question"] == q and r["alpha"] == a)
            row += ("  ✓   " if match["reranked_top1_correct"] else "  ✗   ")
        print(row)

    print("\nClassical top-1 correct (baseline, no reranking):")
    for q in questions:
        baseline = next(r for r in results if r["question"] == q)
        print(f"  {q[:60]}: {'✓' if baseline['classical_top1_correct'] else '✗'}")

gt_results = ground_truth_sweep(ground_truth_queries)
print_ground_truth_summary(gt_results)


# In[ ]:




