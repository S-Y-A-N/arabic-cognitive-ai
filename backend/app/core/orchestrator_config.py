import re

AR_RE = re.compile(r'[\u0600-\u06FF]{3,}')

RESEARCH_KW = ["latest","recent","news","today","2025","2026","current",
               "أحدث","آخر","أخبار","حديث","اليوم","هذا الأسبوع"]
GCC_KW      = ["cbb","sama","uaecb","dfsa","regulation","law","policy","rulebook",
               "vision 2030","رؤية","قانون","نظام","تنظيم","مصرف البحرين",
               "مصرف المركزي","ترخيص","امتثال","compliance","ضوابط","اشتراطات"]
DIALECT_KW  = ["لهجة","dialect","بحريني","خليجي","bahraini","معنى","وايد",
               "حيل","شلون","ترجم","translate","تحليل","تطبيع","code-switching",
               "morphology","فصحى","صرف","جذر"]
EXTRACT_KW  = ["extract","entities","relations","استخرج","كيانات","علاقات"]
REASON_KW   = ["why","how","explain","analyze","compare","لماذا","كيف",
               "اشرح","حلل","قارن","ما الفرق","ما الأسباب"]