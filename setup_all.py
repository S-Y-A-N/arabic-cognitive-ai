"""
ACAI — Master Setup Script
===========================
Run this ONCE to set up everything on your Windows machine.
It copies all files to the right places and generates training data.

Usage:
    cd C:\Users\fatim\arabic-cognitive-ai
    python setup_all.py
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path("C:/Users/fatim/arabic-cognitive-ai")

# ─── Step 1: Create all missing folders ──────────────────────────────────────
print("\n=== Step 1: Creating folders ===")
folders = [
    "eval/data",
    "results",
    "data",
    "paper",
    "models",
    "backend/agents",
    "backend/rag",
    "backend/llm",
    "backend/arabic",
    "backend/knowledge_graph",
    "backend/tools",
    "backend/ingestion",
    "backend/feedback",
    "backend/memory",
    "backend/grc",
    "frontend/src",
    "infrastructure",
    "docker",
]
for folder in folders:
    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ {folder}")

# ─── Step 2: Copy downloaded files to correct locations ──────────────────────
print("\n=== Step 2: Copying files ===")

downloads = Path("C:/Users/fatim/Downloads")
copies = [
    # (source_filename, destination_relative_to_root)
    ("benchmark_harness.py",       "eval/benchmark_harness.py"),
    ("graphrag_impl.py",           "backend/rag/graphrag_impl.py"),
    ("bahraini_qlora_train.py",    "backend/arabic/bahraini_qlora_train.py"),
    ("finetune_bahraini.py",       "finetune_bahraini.py"),
    ("slurm_finetune.sh",          "slurm_finetune.sh"),
    ("experiments_plan.md",        "experiments_plan.md"),
    ("acai_paper_outline.tex",     "paper/acai_paper_outline.tex"),
    ("App_v2.jsx",                 "frontend/src/App.jsx"),
    ("model_config.py",            "backend/model_config.py"),
    ("inference_client.py",        "backend/llm/inference_client.py"),
    ("main_v4.py",                 "backend/main.py"),
    ("langgraph_orchestrator.py",  "backend/agents/langgraph_orchestrator.py"),
    ("advanced_graphrag.py",       "backend/rag/advanced_graphrag.py"),
    ("dialect_specialist.py",      "backend/arabic/dialect_specialist.py"),
    ("grc_module.py",              "backend/grc/grc_module.py"),
    ("tool_registry.py",           "backend/tools/registry.py"),
    ("ingestion_pipeline.py",      "backend/ingestion/pipeline.py"),
    ("feedback_system.py",         "backend/feedback/system.py"),
    ("cognitive_memory.py",        "backend/memory/cognitive_memory.py"),
    ("requirements_v4.txt",        "backend/requirements.txt"),
    ("docker-compose.prod.yml",    "infrastructure/docker-compose.prod.yml"),
    ("docker-compose.yml",         "infrastructure/docker-compose.yml"),
    ("Dockerfile.backend",         "docker/Dockerfile.backend"),
]

for src_name, dest_rel in copies:
    src = downloads / src_name
    dest = ROOT / dest_rel
    if src.exists():
        shutil.copy2(src, dest)
        print(f"  ✅ {src_name} → {dest_rel}")
    else:
        print(f"  ⚠️  NOT FOUND in Downloads: {src_name}")

# ─── Step 3: Create __init__.py files ────────────────────────────────────────
print("\n=== Step 3: Creating __init__.py files ===")
init_dirs = [
    "backend/agents",
    "backend/rag",
    "backend/llm",
    "backend/arabic",
    "backend/knowledge_graph",
    "backend/tools",
    "backend/ingestion",
    "backend/feedback",
    "backend/memory",
    "backend/grc",
]
for d in init_dirs:
    init_file = ROOT / d / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"  ✅ {d}/__init__.py")

# ─── Step 4: Fix memory module name ──────────────────────────────────────────
print("\n=== Step 4: Fixing memory module ===")
src_mem = ROOT / "backend/memory/cognitive_memory.py"
dst_mem = ROOT / "backend/memory/quantum_memory.py"
if src_mem.exists() and not dst_mem.exists():
    shutil.copy2(src_mem, dst_mem)
    print("  ✅ cognitive_memory.py → quantum_memory.py")

# ─── Step 5: Generate Bahraini training data ─────────────────────────────────
print("\n=== Step 5: Generating Bahraini training data ===")

BAHRAINI_SAMPLES = [
    # Daily conversation
    {"instruction": "رد على التحية باللهجة البحرينية", "input": "شلونك؟", "output": "الحمد لله بخير يا خوي، عساك بخير أنت كمان؟ وين كنت هالأيام؟"},
    {"instruction": "اسأل عن الأحوال بالبحريني", "input": "", "output": "شلونك يا خوي؟ شخبارك؟ عساك تمام؟ وايد ما شفتك!"},
    {"instruction": "عبّر عن الفرح بالبحريني", "input": "جيت أول الصف", "output": "والله صج؟ هذا وايد زين! الله يوفقك دايم يا خوي، حيل مبسوط لك!"},
    {"instruction": "كيف تقول 'الآن' بالبحريني؟", "input": "", "output": "بالبحريني نقول 'الحين' بدل 'الآن'. مثلاً: الحين أجي، الحين نروح، الحين خلصنا."},
    {"instruction": "ترجم للبحريني", "input": "أريد أن أذهب إلى السوق", "output": "أبي أروح السوق، أو: بدي أروح السوق الحين."},
    {"instruction": "كيف تعبر عن 'كثير' بالبحريني؟", "input": "", "output": "بالبحريني نستخدم 'وايد' يعني كثير. مثلاً: وايد زين، وايد تعبان، أكلت وايد."},
    {"instruction": "كيف تقول أنك متعب جداً بالبحريني؟", "input": "", "output": "تقدر تقول: حيل تعبان، أو: وايد تعبت اليوم، أو: خلصت طاقتي خالص!"},
    {"instruction": "كيف تقول نعم بالبحريني؟", "input": "", "output": "بالبحريني نقول 'صج' يعني نعم أو صحيح. مثلاً: صج كلامك، صج والله."},
    # Banking
    {"instruction": "اشرح فتح حساب بنكي بالبحريني", "input": "", "output": "عشان تفتح حساب بنكي الحين، لازم تروح للبنك وياك هويتك الشخصية. الموظف راح يطلب منك تعبي استمارة بسيطة وتجيب صورة من الهوية. ما في مشكلة، العملية سريعة وايد."},
    {"instruction": "كيف تحول فلوس بالبحريني؟", "input": "", "output": "تحويل الفلوس الحين سهل حيل. تدخل على التطبيق وتختار تحويل، تكتب رقم الآيبان وتضغط تأكيد. إذا محلي يوصل نفس اليوم."},
    {"instruction": "ما هو مصرف البحرين المركزي؟", "input": "", "output": "مصرف البحرين المركزي هو الجهة التنظيمية للقطاع المصرفي في البحرين. يشرف على البنوك والمؤسسات المالية ويضمن استقرار النظام المالي في المملكة."},
    # Education
    {"instruction": "اشرح أهمية التعليم بالبحريني", "input": "", "output": "التعليم والله شي مهم حيل يا ولدي. اللي يتعلم زين مستقبله أحسن بكثير. الحين السوق يبي ناس متخصصين، إذا تعلمت وتخرجت تلاقي شغل وايد أسهل وراتبك يكون أحسن."},
    {"instruction": "كيف تشجع طالب بالبحريني؟", "input": "", "output": "اشتغل زين يا خوي! أنت وايد ذكي والله. الحين ركّز وذاكر وإن شاء الله تنجح بدرجات عالية. احنا كلنا نشجعك!"},
    # GCC Knowledge
    {"instruction": "ما هي رؤية البحرين 2030؟", "input": "", "output": "رؤية البحرين 2030 هي خطة وطنية تهدف إلى تنويع الاقتصاد البحريني وتقليل الاعتماد على النفط. تركز على تطوير القطاع الخاص وتمكين المواطنين البحرينيين في سوق العمل."},
    {"instruction": "ما هو مجلس التعاون الخليجي؟", "input": "", "output": "مجلس التعاون الخليجي يضم ست دول: البحرين، السعودية، الإمارات، الكويت، قطر، وعُمان. يهدف إلى التكامل الاقتصادي والسياسي بين دول الخليج."},
    # Code-switching
    {"instruction": "استخدم اللهجة البحرينية مع مصطلحات إنجليزية", "input": "كيف أستخدم الـ AI؟", "output": "الـ AI الحين وايد سهل تستخدمه! بس تفتح التطبيق وتكتب سؤالك، وهو راح يجاوبك على طول. وايد زين للدراسة والشغل."},
    {"instruction": "اشرح machine learning بالبحريني", "input": "", "output": "الـ machine learning يعني نعلّم الكمبيوتر يتعلم من البيانات بدون ما نبرمجه خطوة خطوة. مثل ما الطفل يتعلم، الكمبيوتر يشوف أمثلة وايد ويتعلم منها."},
    # More dialect samples
    {"instruction": "كيف تقول 'ليس' بالبحريني؟", "input": "", "output": "بالبحريني نقول 'مب' بدل 'ليس'. مثلاً: مب صحيح، مب أنا، مب الحين."},
    {"instruction": "كيف تطلب شيئاً بلطف بالبحريني؟", "input": "", "output": "تقول: 'لو سمحت'، أو 'من فضلك'، أو 'ممكن تساعدني؟' بالبحريني نقول كمان 'تكفى' يعني من فضلك."},
    {"instruction": "كيف توافق بالبحريني؟", "input": "تروح معي؟", "output": "إي والله، صج أروح! الحين نروح؟ زين خلنا نروح."},
    {"instruction": "كيف ترفض بلطف بالبحريني؟", "input": "تجي معنا؟", "output": "والله عساك تعذرني الحين، مب قادر أجي. إن شاء الله مرة ثانية أكون معاكم."},
]

# Add more GCC policy samples
GCC_SAMPLES = [
    {"instruction": "What does CBB stand for?", "input": "", "output": "CBB stands for the Central Bank of Bahrain (مصرف البحرين المركزي). It is the financial regulatory authority responsible for overseeing banks and financial institutions operating in Bahrain."},
    {"instruction": "Explain SAMA's role", "input": "", "output": "SAMA is the Saudi Central Bank (البنك المركزي السعودي). It regulates and supervises commercial banks in Saudi Arabia, manages monetary policy, and issues banking licenses."},
    {"instruction": "What is Bahrain Vision 2030?", "input": "", "output": "Bahrain Vision 2030 is the national development strategy to diversify Bahrain's economy away from oil dependency. It focuses on developing the private sector, financial services, manufacturing, and tourism while empowering Bahraini citizens in the workforce."},
    {"instruction": "ما هي اشتراطات الترخيص المصرفي في البحرين؟", "input": "", "output": "يشترط مصرف البحرين المركزي للحصول على ترخيص مصرفي توافر الحد الأدنى من رأس المال، والكوادر الإدارية المؤهلة، والهيكل التنظيمي السليم. تتفاوت الاشتراطات حسب نوع الترخيص: بنك تجاري، إسلامي، أو متخصص."},
]

all_samples = BAHRAINI_SAMPLES + GCC_SAMPLES

# Format as ChatML
def to_chatml(item):
    instruction = item.get("instruction", "")
    input_text  = item.get("input", "")
    output      = item.get("output", "")
    user_msg = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
    return {
        "text": (
            f"<|im_start|>system\n"
            f"أنت ACAI، مساعد ذكي متخصص في اللهجة البحرينية والخليجية وشؤون منطقة الخليج. "
            f"أجب بنفس لهجة المستخدم.<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
    }

formatted = [to_chatml(s) for s in all_samples]

# Split 90/10 train/eval
split = int(len(formatted) * 0.9)
train_data = formatted[:split]
eval_data  = formatted[split:]

data_dir = ROOT / "data"
train_path = data_dir / "bahraini_train.jsonl"
eval_path  = data_dir / "bahraini_eval.jsonl"

with open(train_path, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(eval_path, 'w', encoding='utf-8') as f:
    for item in eval_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"  ✅ Generated {len(train_data)} training samples → data/bahraini_train.jsonl")
print(f"  ✅ Generated {len(eval_data)} eval samples    → data/bahraini_eval.jsonl")
print(f"  ℹ️  This is a starter dataset. Add more data for stronger results.")

# ─── Step 6: Print final status ──────────────────────────────────────────────
print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)
print("\n📊 BENCHMARK RESULT (already done):")
print("   Accuracy: 87.5% on ABBL — save this as baseline!")
print("\n📁 Files are in place.")
print("\n🚀 Next steps:")
print("\n1. Activate venv and run Bahraini benchmark:")
print("   cd C:\\Users\\fatim\\arabic-cognitive-ai\\backend")
print("   venv\\Scripts\\activate")
print("   cd ..")
print("   python eval\\benchmark_harness.py --benchmark bahraini --model qwen2.5:14b-instruct-q4_K_M")
print("\n2. Upload data to lab:")
print("   scp data\\bahraini_train.jsonl user151@hayrat.uob.edu.bh:/data/datasets/user151/acai/data/")
print("   scp data\\bahraini_eval.jsonl  user151@hayrat.uob.edu.bh:/data/datasets/user151/acai/data/")
print("\n3. SSH into lab and fix conda env:")
print("   ssh user151@hayrat.uob.edu.bh")
print("   conda deactivate")
print("   conda env remove -n acai-train -y")
print("   conda create --prefix /data/datasets/user151/acai/conda-env python=3.11 -y")
print("   conda activate /data/datasets/user151/acai/conda-env")
print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
print("   pip install transformers peft trl bitsandbytes accelerate datasets sentencepiece")
print("\n4. Submit training job:")
print("   sbatch /data/datasets/user151/acai/slurm_finetune.sh")
