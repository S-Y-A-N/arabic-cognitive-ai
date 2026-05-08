from app.services.orchestrator import orchestrate

BAHRAINI_MARKERS = ["الحين","وايد","حيل","شلونك","خوي","صج","مب","عساك","زين","تره","باكر"]
MSA_MARKERS      = ["كيف حالك","أنا بخير","شكراً لك","ينبغي","يجب أن","لا أعلم"]

async def run_dcr_eval() -> dict:
    """Measure Dialect Control Rate and MSA Leak Rate."""
    prompts = [
        "أجب باللهجة البحرينية فقط: كيف حالك؟",
        "اشرح كيف تفتح حساب بنكي باللهجة البحرينية",
        "قل لي أنك لا تعرف الإجابة باللهجة البحرينية",
    ]
    dcr_scores, mlr_scores = [], []
    for prompt in prompts:
        result = await orchestrate(prompt, mode="single:lughawi")
        resp   = result["answer"].lower()
        dcr = sum(1 for m in BAHRAINI_MARKERS if m in resp) / len(BAHRAINI_MARKERS)
        mlr = sum(1 for m in MSA_MARKERS     if m in resp) / len(MSA_MARKERS)
        dcr_scores.append(dcr); mlr_scores.append(mlr)
    return {
        "avg_dcr":   round(sum(dcr_scores)/len(dcr_scores), 3),
        "avg_mlr":   round(sum(mlr_scores)/len(mlr_scores), 3),
        "dcr_per":   dcr_scores,
        "mlr_per":   mlr_scores,
    }


async def run_memory_experiment(questions: list) -> dict:
    """
    Before/after memory experiment.
    Clears memory context for 'before' queries, uses accumulated memory for 'after'.
    Returns comparison — this is your paper's Table 2.
    """
    results = []
    # 'without_memory' pass — clear memory context by using fresh queries
    for q in questions[:8]:
        r = await orchestrate(q, mode="auto")
        results.append({"question": q[:80],
                         "without_memory_latency": r["latency_ms"],
                         "without_memory_answer":  r["answer"][:200]})

    # 'with_memory' pass — memory now has context from above queries
    for i, q in enumerate(questions[:8]):
        r = await orchestrate(q, mode="auto")
        results[i]["with_memory_latency"]   = r["latency_ms"]
        results[i]["with_memory_answer"]    = r["answer"][:200]
        results[i]["memory_was_available"]  = r["memory_used"]

    return {
        "results":  results,
        "summary":  memory.experiment_summary(),
        "note":     "Compare without_memory_answer vs with_memory_answer for quality improvement",
    }