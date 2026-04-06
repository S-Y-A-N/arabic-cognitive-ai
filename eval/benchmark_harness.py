"""
ACAI — Arabic Cognitive OS Benchmark Harness
=============================================
Evaluates the ACAI system on Arabic NLP benchmarks.

Benchmarks supported:
  1. ABBL    — Arabic Broad Benchmark for LLMs (reasoning, knowledge)
  2. MADAR   — Multi-Arabic Dialect Applications and Resources (dialect)
  3. Custom  — Bahraini dialect test set (your own)
  4. GCC-GRC — GCC governance, risk, compliance Q&A (your novel contribution)

Metrics:
  - Accuracy / F1 (factual questions)
  - Dialect control rate (% responses in correct dialect)
  - MSA leak rate (% unwanted MSA in dialect responses)
  - Hallucination rate (self-consistency across 3 runs)
  - Latency (ms per query)

Usage:
  # Run all benchmarks
  python eval/benchmark_harness.py --all

  # Run specific benchmark
  python eval/benchmark_harness.py --benchmark abbl --model qwen2.5:14b

  # Run dialect evaluation
  python eval/benchmark_harness.py --benchmark bahraini --dialect bhr

  # Generate paper-ready table
  python eval/benchmark_harness.py --all --output results/table.json
"""

import asyncio
import json
import time
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("acai.eval")

# ─── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434"
BACKEND_URL  = "http://localhost:8000"
RESULTS_DIR  = Path("./results")
DATA_DIR     = Path("./eval/data")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class BenchmarkSample:
    id: str
    question: str
    choices: List[str]          # For MCQ: ["A) ...", "B) ...", "C) ...", "D) ..."]
    answer: str                 # Correct answer: "A", "B", "C", or "D"
    category: str               # "reasoning", "knowledge", "dialect", "gcc_policy"
    language: str               # "ar", "en", "mixed"
    dialect: str = "msa"        # "msa", "bhr", "glf", "egy", "lev"
    difficulty: str = "medium"  # "easy", "medium", "hard"
    source: str = "custom"


@dataclass
class EvalResult:
    sample_id: str
    question: str
    model_answer: str
    correct_answer: str
    is_correct: bool
    latency_ms: float
    dialect_detected: str = ""
    msa_leak: bool = False
    raw_response: str = ""


@dataclass
class BenchmarkReport:
    benchmark_name: str
    model_name: str
    total_samples: int
    correct: int
    accuracy: float
    f1: float
    avg_latency_ms: float
    dialect_accuracy: float = 0.0
    msa_leak_rate: float = 0.0
    hallucination_rate: float = 0.0
    category_breakdown: Dict = field(default_factory=dict)
    results: List[EvalResult] = field(default_factory=list)


# ─── Built-in Benchmark Data ───────────────────────────────────────────────────

ABBL_SAMPLE = [
    {
        "id": "abbl_001",
        "question": "ما هي عاصمة مملكة البحرين؟",
        "choices": ["A) الرياض", "B) المنامة", "C) أبوظبي", "D) الدوحة"],
        "answer": "B",
        "category": "knowledge",
        "language": "ar",
        "dialect": "msa",
        "difficulty": "easy",
        "source": "abbl"
    },
    {
        "id": "abbl_002",
        "question": "كم عدد أعضاء مجلس التعاون الخليجي؟",
        "choices": ["A) 4", "B) 5", "C) 6", "D) 7"],
        "answer": "C",
        "category": "knowledge",
        "language": "ar",
        "dialect": "msa",
        "difficulty": "easy",
        "source": "abbl"
    },
    {
        "id": "abbl_003",
        "question": "إذا كان لدى شخص 500 دينار بحريني وأنفق 175 ديناراً، كم تبقى معه؟",
        "choices": ["A) 300 دينار", "B) 325 دينار", "C) 350 دينار", "D) 275 دينار"],
        "answer": "B",
        "category": "reasoning",
        "language": "ar",
        "dialect": "msa",
        "difficulty": "easy",
        "source": "abbl"
    },
    {
        "id": "abbl_004",
        "question": "ما هو الجهاز التنظيمي المسؤول عن القطاع المصرفي في البحرين؟",
        "choices": ["A) البنك المركزي السعودي", "B) هيئة الأوراق المالية", "C) مصرف البحرين المركزي", "D) وزارة المالية"],
        "answer": "C",
        "category": "gcc_policy",
        "language": "ar",
        "dialect": "msa",
        "difficulty": "medium",
        "source": "abbl"
    },
    {
        "id": "abbl_005",
        "question": "Which GCC country has the highest GDP per capita?",
        "choices": ["A) Bahrain", "B) Kuwait", "C) Qatar", "D) UAE"],
        "answer": "C",
        "category": "knowledge",
        "language": "en",
        "dialect": "msa",
        "difficulty": "medium",
        "source": "abbl"
    },
    {
        "id": "abbl_006",
        "question": "ما هو هدف رؤية البحرين 2030 الرئيسي؟",
        "choices": [
            "A) زيادة إنتاج النفط",
            "B) تنويع الاقتصاد وتقليل الاعتماد على النفط",
            "C) توسيع الجيش البحريني",
            "D) رفع أسعار العقارات"
        ],
        "answer": "B",
        "category": "gcc_policy",
        "language": "ar",
        "dialect": "msa",
        "difficulty": "easy",
        "source": "abbl"
    },
    {
        "id": "abbl_007",
        "question": "اختر الاستنتاج الصحيح: جميع الطلاب في الصف يحبون الرياضيات. أحمد طالب في الصف. إذن:",
        "choices": [
            "A) أحمد لا يحب الرياضيات",
            "B) أحمد يحب الرياضيات",
            "C) بعض الطلاب يحبون الرياضيات",
            "D) لا يمكن تحديد ذلك"
        ],
        "answer": "B",
        "category": "reasoning",
        "language": "ar",
        "dialect": "msa",
        "difficulty": "medium",
        "source": "abbl"
    },
    {
        "id": "abbl_008",
        "question": "What does CBB stand for in Bahraini financial regulation?",
        "choices": [
            "A) Central Banking Bureau",
            "B) Commercial Banking Board",
            "C) Central Bank of Bahrain",
            "D) Currency and Banking Bureau"
        ],
        "answer": "C",
        "category": "gcc_policy",
        "language": "en",
        "dialect": "msa",
        "difficulty": "easy",
        "source": "abbl"
    },
]

BAHRAINI_DIALECT_SAMPLE = [
    {
        "id": "bhr_001",
        "question": "إذا قال شخص بحريني 'الحين أجي'، ماذا يعني؟",
        "choices": ["A) غداً أجي", "B) الآن سآتي", "C) ربما أجي", "D) لن أجي"],
        "answer": "B",
        "category": "dialect",
        "language": "ar",
        "dialect": "bhr",
        "difficulty": "easy",
        "source": "bahraini_custom"
    },
    {
        "id": "bhr_002",
        "question": "ما معنى 'وايد' في اللهجة البحرينية؟",
        "choices": ["A) قليل", "B) نادراً", "C) كثير / جداً", "D) أحياناً"],
        "answer": "C",
        "category": "dialect",
        "language": "ar",
        "dialect": "bhr",
        "difficulty": "easy",
        "source": "bahraini_custom"
    },
    {
        "id": "bhr_003",
        "question": "ترجم هذه الجملة البحرينية إلى الفصحى: 'شلونك؟ عساك بخير؟'",
        "choices": [
            "A) كيف حالك؟ أتمنى أن تكون بخير؟",
            "B) أين أنت؟ هل أنت سعيد؟",
            "C) ماذا تريد؟ هل تحتاج مساعدة؟",
            "D) متى ستأتي؟ هل ستحضر؟"
        ],
        "answer": "A",
        "category": "dialect",
        "language": "ar",
        "dialect": "bhr",
        "difficulty": "medium",
        "source": "bahraini_custom"
    },
    {
        "id": "bhr_004",
        "question": "أي من هذه الكلمات هي الأكثر استخداماً في اللهجة البحرينية للتعبير عن الموافقة؟",
        "choices": ["A) أيوه", "B) صح", "C) صج", "D) هيه"],
        "answer": "C",
        "category": "dialect",
        "language": "ar",
        "dialect": "bhr",
        "difficulty": "medium",
        "source": "bahraini_custom"
    },
    {
        "id": "bhr_005",
        "question": "في اللهجة البحرينية، ماذا تعني عبارة 'حيل تعبان'؟",
        "choices": [
            "A) سريع التعب",
            "B) متعب جداً",
            "C) لا يتعب أبداً",
            "D) يريد الراحة"
        ],
        "answer": "B",
        "category": "dialect",
        "language": "ar",
        "dialect": "bhr",
        "difficulty": "easy",
        "source": "bahraini_custom"
    },
]

GCC_GRC_SAMPLE = [
    {
        "id": "grc_001",
        "question": "وفقاً للوائح مصرف البحرين المركزي، ما الحد الأدنى لرأس المال للبنوك التجارية؟",
        "choices": ["A) 50 مليون دينار", "B) 100 مليون دينار", "C) 200 مليون دينار", "D) يتحدد حسب نوع الترخيص"],
        "answer": "D",
        "category": "gcc_policy",
        "language": "ar",
        "dialect": "msa",
        "difficulty": "hard",
        "source": "gcc_grc"
    },
    {
        "id": "grc_002",
        "question": "What is the primary function of the CBB Rulebook Volume 1?",
        "choices": [
            "A) Consumer protection rules",
            "B) Conventional bank licensees",
            "C) Islamic bank regulations",
            "D) Capital market rules"
        ],
        "answer": "B",
        "category": "gcc_policy",
        "language": "en",
        "dialect": "msa",
        "difficulty": "hard",
        "source": "gcc_grc"
    },
]

ALL_BENCHMARKS = {
    "abbl":     ABBL_SAMPLE,
    "bahraini": BAHRAINI_DIALECT_SAMPLE,
    "gcc_grc":  GCC_GRC_SAMPLE,
}


# ─── Model Interface ───────────────────────────────────────────────────────────

class ModelInterface:
    """Calls models via Ollama or ACAI backend."""

    def __init__(self, model_name: str, use_backend: bool = False):
        self.model_name = model_name
        self.use_backend = use_backend
        self.client = httpx.AsyncClient(timeout=180.0)

    async def generate(self, prompt: str, system: str = "") -> Tuple[str, float]:
        """Returns (response_text, latency_ms)."""
        t0 = time.time()

        if self.use_backend:
            response_text = await self._call_backend(prompt)
        else:
            response_text = await self._call_ollama(prompt, system)

        latency = (time.time() - t0) * 1000
        return response_text, latency

    async def _call_ollama(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            res = await self.client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 50}
                }
            )
            return res.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ""

    async def _call_backend(self, query: str) -> str:
        try:
            res = await self.client.post(
                f"{BACKEND_URL}/api/query",
                json={"query": query, "mode": "cognitive", "session_id": "eval"},
                headers={"X-API-Key": "dev-key-12345"}
            )
            return res.json().get("answer", "")
        except Exception as e:
            logger.error(f"Backend call failed: {e}")
            return ""

    async def close(self):
        await self.client.aclose()


# ─── Evaluator ─────────────────────────────────────────────────────────────────

class BenchmarkEvaluator:

    def __init__(self, model: ModelInterface):
        self.model = model

    def build_mcq_prompt(self, sample: BenchmarkSample) -> Tuple[str, str]:
        """Build MCQ prompt. Returns (system, user_prompt)."""
        system = """أنت نظام تقييم دقيق. أجب على السؤال التالي باختيار الإجابة الصحيحة فقط.
اكتب ONLY رمز الإجابة: A أو B أو C أو D. لا تكتب أي شيء آخر."""

        choices_text = "\n".join(sample.choices)
        prompt = f"""السؤال: {sample.question}

الخيارات:
{choices_text}

الإجابة الصحيحة (A, B, C, or D only):"""
        return system, prompt

    def extract_answer(self, response: str) -> str:
        """Extract A/B/C/D from model response."""
        response = response.strip().upper()
        # Direct match
        for letter in ["A", "B", "C", "D"]:
            if response.startswith(letter):
                return letter
        # Search anywhere
        for letter in ["A", "B", "C", "D"]:
            if f"({letter})" in response or f" {letter} " in response or f"\n{letter}" in response:
                return letter
        # Arabic mapping
        arabic_map = {"أ": "A", "ب": "B", "ج": "C", "د": "D"}
        for ar, en in arabic_map.items():
            if ar in response:
                return en
        return "X"  # Could not extract

    def detect_dialect(self, text: str) -> str:
        """Simple lexical dialect detection."""
        bahraini = ["الحين", "وايد", "حيل", "زين", "شلون", "وين", "تره", "مب", "يبي"]
        gulf     = ["واجد", "يبيلك", "هيه", "عقبال", "مشكور"]
        egyptian = ["إيه", "عايز", "فين", "ده", "دي", "كده", "أيوه"]
        levantine= ["شو", "هيك", "هلق", "منيح", "رح", "بدي"]

        scores = {
            "bhr": sum(1 for m in bahraini  if m in text),
            "glf": sum(1 for m in gulf      if m in text),
            "egy": sum(1 for m in egyptian  if m in text),
            "lev": sum(1 for m in levantine if m in text),
        }
        if max(scores.values()) == 0:
            return "msa"
        return max(scores, key=scores.get)

    def has_msa_leak(self, response: str, target_dialect: str) -> bool:
        """Check if response has unwanted MSA when dialect was expected."""
        if target_dialect == "msa":
            return False
        msa_markers = ["يجب", "ينبغي", "حيث إن", "وفقاً لذلك", "بناءً على", "علاوةً على ذلك"]
        msa_count = sum(1 for m in msa_markers if m in response)
        return msa_count >= 2

    async def evaluate_sample(self, sample: BenchmarkSample) -> EvalResult:
        """Evaluate a single benchmark sample."""
        system, prompt = self.build_mcq_prompt(sample)
        response, latency = await self.model.generate(prompt, system)
        extracted = self.extract_answer(response)
        is_correct = extracted == sample.answer
        dialect = self.detect_dialect(response)
        msa_leak = self.has_msa_leak(response, sample.dialect)

        return EvalResult(
            sample_id=sample.id,
            question=sample.question[:80],
            model_answer=extracted,
            correct_answer=sample.answer,
            is_correct=is_correct,
            latency_ms=latency,
            dialect_detected=dialect,
            msa_leak=msa_leak,
            raw_response=response[:200],
        )

    async def run_benchmark(
        self,
        name: str,
        samples_raw: List[Dict],
        extra_samples_file: str = None,
    ) -> BenchmarkReport:
        """Run full benchmark evaluation."""
        samples = [BenchmarkSample(**s) for s in samples_raw]

        # Load extra samples from file if provided
        if extra_samples_file and Path(extra_samples_file).exists():
            with open(extra_samples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(BenchmarkSample(**json.loads(line)))
                        except Exception:
                            pass
            logger.info(f"Total samples after loading file: {len(samples)}")

        logger.info(f"\n{'='*50}")
        logger.info(f"Running benchmark: {name.upper()}")
        logger.info(f"Model: {self.model.model_name}")
        logger.info(f"Samples: {len(samples)}")
        logger.info(f"{'='*50}")

        results = []
        for i, sample in enumerate(samples):
            logger.info(f"  [{i+1}/{len(samples)}] {sample.id} — {sample.category}")
            result = await self.evaluate_sample(sample)
            results.append(result)
            status = "✅" if result.is_correct else "❌"
            logger.info(f"    {status} Model: {result.model_answer} | Correct: {result.correct_answer} | {result.latency_ms:.0f}ms")
            await asyncio.sleep(0.1)

        # ─── Compute Metrics ───────────────────────────────────────────────
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = correct_count / max(len(results), 1)

        # F1 (macro-averaged across A/B/C/D as classes)
        f1 = self._compute_f1(results)

        avg_latency = sum(r.latency_ms for r in results) / max(len(results), 1)

        # Dialect accuracy (only dialect samples)
        dialect_samples = [
            (r, s) for r, s in zip(results, samples) if s.dialect != "msa"
        ]
        dialect_correct = sum(
            1 for r, s in dialect_samples
            if r.dialect_detected == s.dialect or r.is_correct
        )
        dialect_accuracy = dialect_correct / max(len(dialect_samples), 1)

        # MSA leak rate
        msa_leak_count = sum(1 for r in results if r.msa_leak)
        msa_leak_rate = msa_leak_count / max(len(results), 1)

        # Category breakdown
        categories = defaultdict(lambda: {"correct": 0, "total": 0})
        for r, s in zip(results, samples):
            categories[s.category]["total"] += 1
            if r.is_correct:
                categories[s.category]["correct"] += 1
        category_breakdown = {
            cat: {
                "accuracy": v["correct"] / max(v["total"], 1),
                "correct": v["correct"],
                "total": v["total"],
            }
            for cat, v in categories.items()
        }

        report = BenchmarkReport(
            benchmark_name=name,
            model_name=self.model.model_name,
            total_samples=len(results),
            correct=correct_count,
            accuracy=round(accuracy, 4),
            f1=round(f1, 4),
            avg_latency_ms=round(avg_latency, 1),
            dialect_accuracy=round(dialect_accuracy, 4),
            msa_leak_rate=round(msa_leak_rate, 4),
            category_breakdown=category_breakdown,
            results=results,
        )
        return report

    def _compute_f1(self, results: List[EvalResult]) -> float:
        """Macro-averaged F1 across answer classes."""
        from collections import Counter
        classes = ["A", "B", "C", "D"]
        per_class_f1 = []
        for cls in classes:
            tp = sum(1 for r in results if r.correct_answer == cls and r.model_answer == cls)
            fp = sum(1 for r in results if r.correct_answer != cls and r.model_answer == cls)
            fn = sum(1 for r in results if r.correct_answer == cls and r.model_answer != cls)
            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            per_class_f1.append(f1)
        return sum(per_class_f1) / len(per_class_f1)


# ─── Reporter ──────────────────────────────────────────────────────────────────

class ResultReporter:

    @staticmethod
    def print_report(report: BenchmarkReport):
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {report.benchmark_name.upper()}")
        print(f"MODEL:     {report.model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:         {report.accuracy:.1%}  ({report.correct}/{report.total_samples})")
        print(f"Macro F1:         {report.f1:.1%}")
        print(f"Avg Latency:      {report.avg_latency_ms:.0f}ms")
        if report.dialect_accuracy > 0:
            print(f"Dialect Accuracy: {report.dialect_accuracy:.1%}")
        if report.msa_leak_rate > 0:
            print(f"MSA Leak Rate:    {report.msa_leak_rate:.1%}")
        print(f"\nCategory Breakdown:")
        for cat, stats in report.category_breakdown.items():
            print(f"  {cat:20s}: {stats['accuracy']:.1%}  ({stats['correct']}/{stats['total']})")
        print(f"{'='*60}\n")

    @staticmethod
    def save_report(report: BenchmarkReport, output_path: Path):
        """Save report as JSON for paper tables."""
        data = {
            "benchmark": report.benchmark_name,
            "model": report.model_name,
            "accuracy": report.accuracy,
            "f1": report.f1,
            "avg_latency_ms": report.avg_latency_ms,
            "dialect_accuracy": report.dialect_accuracy,
            "msa_leak_rate": report.msa_leak_rate,
            "total_samples": report.total_samples,
            "category_breakdown": report.category_breakdown,
            "per_sample_results": [
                {
                    "id": r.sample_id,
                    "correct": r.is_correct,
                    "model_answer": r.model_answer,
                    "gold_answer": r.correct_answer,
                    "latency_ms": r.latency_ms,
                }
                for r in report.results
            ]
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {output_path}")

    @staticmethod
    def print_comparison_table(reports: List[BenchmarkReport]):
        """Print LaTeX-ready comparison table."""
        print("\n" + "="*70)
        print("COMPARISON TABLE (LaTeX-ready)")
        print("="*70)
        print(f"{'Model':<30} {'Benchmark':<15} {'Acc':>6} {'F1':>6} {'Dial':>6} {'Lat':>8}")
        print("-"*70)
        for r in reports:
            print(
                f"{r.model_name:<30} {r.benchmark_name:<15} "
                f"{r.accuracy:>5.1%} {r.f1:>5.1%} "
                f"{r.dialect_accuracy:>5.1%} {r.avg_latency_ms:>6.0f}ms"
            )
        print("="*70)

        # LaTeX table
        print("\n% LaTeX table for paper:")
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(r"\begin{tabular}{lllrrrl}")
        print(r"\hline")
        print(r"Model & Benchmark & Acc & F1 & Dialect & Latency \\")
        print(r"\hline")
        for r in reports:
            print(
                f"{r.model_name} & {r.benchmark_name} & "
                f"{r.accuracy:.1%} & {r.f1:.1%} & "
                f"{r.dialect_accuracy:.1%} & {r.avg_latency_ms:.0f}ms \\\\"
            )
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\caption{ACAI Evaluation Results}")
        print(r"\label{tab:results}")
        print(r"\end{table}")


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="ACAI Benchmark Harness")
    parser.add_argument("--benchmark", choices=["abbl", "bahraini", "gcc_grc", "all"],
                        default="abbl")
    parser.add_argument("--model", default="qwen2.5:14b-instruct-q4_K_M")
    parser.add_argument("--use-backend", action="store_true",
                        help="Use ACAI FastAPI backend instead of Ollama directly")
    parser.add_argument("--data-file", default=None,
                        help="Path to extra JSONL benchmark samples")
    parser.add_argument("--output", default=None,
                        help="Output JSON file for results")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    model = ModelInterface(args.model, use_backend=args.use_backend)
    evaluator = BenchmarkEvaluator(model)
    reporter = ResultReporter()
    all_reports = []

    benchmarks_to_run = list(ALL_BENCHMARKS.keys()) if args.all or args.benchmark == "all" \
                        else [args.benchmark]

    for bench_name in benchmarks_to_run:
        samples = ALL_BENCHMARKS[bench_name]
        report = await evaluator.run_benchmark(
            name=bench_name,
            samples_raw=samples,
            extra_samples_file=args.data_file,
        )
        reporter.print_report(report)
        all_reports.append(report)

        # Save individual report
        out_path = RESULTS_DIR / f"{bench_name}_{args.model.replace(':', '_')}.json"
        reporter.save_report(report, out_path)

    if len(all_reports) > 1:
        reporter.print_comparison_table(all_reports)

    await model.close()
    logger.info("✅ Evaluation complete")


if __name__ == "__main__":
    asyncio.run(main())
