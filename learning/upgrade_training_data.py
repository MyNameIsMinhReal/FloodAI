# -*- coding: utf-8 -*-
"""
learning/upgrade_training_data.py
=================================
Nâng cấp + mở rộng training data cho FloodAgent LLM.

3 chức năng:
  1. --gen     : Gen thêm nhiều mẫu với ngôn ngữ TỰ NHIÊN HƠN (slang, lỗi
                 chính tả, câu lủng củng, tiếng địa phương, cảm xúc)
                 GỘP vào data hiện có (không mất cũ — đã backup tự động).
  2. --export  : Xuất hội thoại thật từ AgentMemory (agent/_agent_memory/
                 long_term_memory.json) thành training samples.
  3. --all     : Cả 2 ở trên + ghi đè file tổng hợp.

Mục đích: model học data "thật hơn" thay vì chỉ template sạch — giúp
khái quát tốt khi gặp cách nói tự nhiên của người dùng thật.

Chạy:
  python -m learning.upgrade_training_data --all
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent / "training_data"
TRAIN_FILE = DATA_DIR / "flood_conversations.jsonl"
EVAL_FILE  = DATA_DIR / "flood_conversations_eval.jsonl"
MEM_FILE   = Path(__file__).parent.parent / "agent" / "_agent_memory" / "long_term_memory.json"

SYSTEM_PROMPT = (
    "Bạn là FloodAgent — AI phân tích lũ lụt thông minh, hỗ trợ tiếng Việt và tiếng Anh.\n"
    "Nhiệm vụ: đọc kết quả phân tích lũ và phản hồi người dùng tự nhiên, chính xác.\n\n"
    "Khi người dùng gửi tin nhắn, hãy trả lời theo đúng định dạng sau:\n"
    "INTENT: <intent>\n"
    "DEPTH_HINT: <số cm hoặc null>\n"
    "LEVEL_HINT: <ANKLE/KNEE/WAIST/CHEST/SUBMERGED/NO_FLOOD hoặc null>\n\n"
    "<phản hồi tự nhiên bằng tiếng Việt>\n\n"
    "Các intent hợp lệ: confirm | increase | decrease | no_flood | "
    "rerun | simulate | status | help | calibrate | unknown"
)

LEVELS = {
    "NO_FLOOD":  ("Không ngập",                  0),
    "PUDDLE":    ("Vũng nước nhỏ (<15cm)",        10),
    "ANKLE":     ("Ngập mắt cá (15-40cm)",        25),
    "KNEE":      ("Ngập đầu gối (40-70cm)",       55),
    "WAIST":     ("Ngập ngang hông (70-120cm)",   90),
    "CHEST":     ("Ngập ngang ngực (120-200cm)",  150),
    "SUBMERGED": ("Ngập hoàn toàn (>200cm)",      250),
}

INTENTS = ["confirm", "increase", "decrease", "no_flood", "rerun",
           "simulate", "status", "help", "calibrate", "unknown"]

# ── Các mẫu câu "thật" — tự nhiên, lủng củng, văn nói ─────────────────────────

# Các biến thể đa dạng theo intent (giàu cảm xúc, địa phương, lỗi chính tả)
REAL_PHRASES: Dict[str, List[str]] = {
    "confirm": [
        "Đúng rồi bạn, chuẩn không cần chỉnh luôn",
        "ừ đúng đó, mình thấy khớp với thực tế",
        "chính xác 100%, nhà mình ngập đúng mức vậy nè",
        "dạ đúng rồi ạ, cám ơn bạn nhiều nha",
        "ủa chuẩn thiệt hả? ok luôn",
        "hợp lý á, theo dõi vậy được rồi",
        "đúng luôn ông ơi, y chang bên ngoài",
        "yeah chính xác, để mình ghi nhận kết quả này",
        "đúng vậy, mực nước nhà mình tầm đó á",
        "okê khớp với những gì mình thấy",
    ],
    "increase": [
        "nước dâng lên nhanh quá bạn ơi, chắc phải sâu hơn đó",
        "ê nước lên rồi, chừng 40 phân thêm gì đó",
        "đang lên à? thấy ngập tới gối rồi nè",
        "nước cao hơn lúc nãy nhiều, khoảng {depth_new} phân á",
        "dạo này mưa hoài nước càng lúc càng dâng",
        "nước đang lớn mạnh quá, chắc trên {depth_new} cm rồi",
        "sao nước lên dữ vậy? giờ chắc ngập tới {level_hint}",
        "hình như nước đang dâng lên không ngừng",
        "nước cao hơn rồi, phải cập nhật lên mức cao hơn chứ",
        "nó đang lên từng giờ, báo mức cao lên đi ông",
    ],
    "decrease": [
        "nước rút rồi bạn, giảm xuống thôi",
        "chỗ mình nước đang hạ, chắc còn tầm {depth_new} phân",
        "có vẻ nước đã rút bớt so với lúc sáng",
        "nước xuống rồi, khỏi lo nữa",
        "nước đang tụt dần, để mình xác nhận mức giảm",
        "thấy nước rút nhanh ghê, giảm mức đi bạn",
        "nước hạ rồi, còn ngập tới đầu gối thôi hà",
        "nước rút bớt khoảng {depth_new} cm rồi đó",
        "không còn dâng nữa, đang giảm đây",
        "nước xuống rồi nha, cập nhật lại đi",
    ],
    "no_flood": [
        "mà hình như chỗ này không ngập, đường khô ráo mà",
        "khoan đã, nhà mình đâu có nước đâu",
        "hình như ông nhầm rồi, chỗ này khô nguyên",
        "làm gì có ngập ở đây, xe chạy bon bon mà",
        "nhìn kỹ lại đi, không thấy nước gì hết",
        "chắc ảnh chụp nhầm chỗ rồi, ngoài này khô lắm",
        "đâu có ngập đâu má, đường ráo hoảnh",
        "không có nước nha, báo nhầm rồi",
        "ảnh hưởng gì đâu, trời nắng chang chang mà",
        "thôi khỏi, đây không phải khu ngập đâu",
    ],
    "status": [
        "tình hình hiện tại sao rồi?",
        "giờ nước ngập tới đâu vậy bạn?",
        "cho mình xem diễn biến mực nước hiện tại đi",
        "hiện tại đang ở mức nào?",
        "cập nhật tình trạng ngập hiện giờ giúp mình",
    ],
    "help": [
        "bạn hướng dẫn mình dùng thế nào?",
        "app này xài sao vậy?",
        "mình cần làm gì để phân tích ảnh?",
        "giúp mình với, mình không biết bắt đầu từ đâu",
        "bạn ơi chỉ mình cách dùng đi",
    ],
    "calibrate": [
        "kết quả hơi lệch so với thực tế, chỉnh lại giúp mình",
        "mực nước đo bị sai chút, calibrate lại đi bạn",
        "hệ thống nó báo cao hơn thực tế á, sửa giúp",
        "chỉnh hiệu chuẩn lại nha, thấy hơi chênh",
        "đo lại độ sâu giúp mình, số hơi vô lý",
    ],
    "unknown": [
        "hà hà, kể chuyện cười cho mình nghe đi",
        "bạn tên gì thế?",
        "trời hôm nay đẹp quá ha",
        "bạn là ai vậy?",
        "chán quá, trò chuyện với mình đi",
    ],
}

# Các placeholder hội thoại chưa có context ảnh (mở đầu chat)
NO_CONTEXT_REAL = [
    "Chào bạn, mình muốn hỏi về lụt ở khu vực mình",
    "Bạn ơi kiểm tra giúp ảnh gửi kèm",
    "Mình mới tới, dùng app này sao vậy?",
    "ô bạn, phân tích giúp mình bức ảnh này với",
    "Xin chào, giúp mình với ạ",
]


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _make_example(user_text: str, intent: str,
                  level: Optional[str] = None, depth: Optional[int] = None) -> Dict:
    """Tạo 1 training sample nhất quán với định dạng trainer đang dùng."""
    user_content = (
        f"[Kết quả phân tích hiện tại]\n"
        f"- Mức lũ   : {level or 'Không xác định'}\n"
        f"- Độ sâu   : {depth if depth is not None else 0}cm\n"
        f"- Độ tin cậy: {random.randint(40, 90)}%\n\n"
        f"Người dùng nói: \"{user_text}\""
    )
    level_hint = level if level else "null"
    depth_hint = str(depth) if depth is not None else "null"
    assistant_content = (
        f"INTENT: {intent}\n"
        f"DEPTH_HINT: {depth_hint}\n"
        f"LEVEL_HINT: {level_hint}\n\n"
        f"{_natural_reply(intent, level)}"
    )
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _natural_reply(intent: str, level: Optional[str]) -> str:
    """Trả lời tự nhiên, văn nói — không cứng nhắc."""
    level_vi = LEVELS.get(level, ("",))[0] if level else ""
    replies = {
        "confirm": [
            f"Tuyệt vời! Mình ghi nhận xác nhận của bạn về mức {level_vi}.",
            "Cảm ơn bạn đã xác nhận! Kết quả này rất đáng tin cậy.",
            "Ok chuẩn rồi, mình sẽ lưu thông tin này lại.",
        ],
        "increase": [
            f"Nước đang dâng lên — bạn báo đúng. Mình cập nhật lên mức cao hơn {level_vi}.",
            "Đã ghi nhận mức nước tăng. Bạn nên chủ động di chuyển lên nơi cao hơn nhé.",
            "Cảnh báo mực nước dâng! Mình đã cập nhật mức mới.",
        ],
        "decrease": [
            f"Nước đang rút — tin tốt! Mình hạ mức xuống {level_vi}.",
            "Đã cập nhật giảm mực nước. Tình hình đang cải thiện.",
            "Nước rút rồi, mình điều chỉnh kết quả xuống.",
        ],
        "no_flood": [
            "Rõ bạn! Chỗ này không ngập, mình bỏ qua ảnh hưởng ngập lụt.",
            "Cảm ơn phản hồi — đây không phải khu vực ngập.",
            "Ok không ngập, mình ghi nhận lại.",
        ],
        "status": [
            "Hiện tại tình hình mực nước đang ở mức trung bình, cần theo dõi tiếp.",
            "Mực nước hiện tại đang ổn định, chưa có biến động lớn.",
        ],
        "help": [
            "Mình sẽ hướng dẫn bạn từng bước: đầu tiên upload ảnh khu vực bạn muốn kiểm tra.",
            "Để bắt đầu, bạn gửi ảnh chụp khu vực đang ngập để mình phân tích nhé.",
        ],
        "calibrate": [
            "Mình đã ghi nhận độ lệch và sẽ hiệu chuẩn lại kết quả.",
            "Ok, mình sẽ điều chỉnh hiệu chuẩn để chính xác hơn.",
        ],
        "rerun": [
            "Được, mình sẽ phân tích lại bức ảnh này với tham số mới.",
            "Rerun ngay! Mình chạy lại pipeline để cập nhật kết quả.",
        ],
        "simulate": [
            "Mình đã chạy mô phỏng — theo kịch bản hiện tại nước có thể dâng thêm vài cm.",
            "Simulate xong, mức độ rủi ro hiện ở mức cần chú ý.",
        ],
        "unknown": [
            "Mình là FloodAgent, chuyên phân tích ngập lụt. Bạn cần mình giúp gì về lũ không?",
            "Bạn hãy kể cho mình nghe tình hình khu vực của bạn nhé?",
        ],
    }
    return random.choice(replies.get(intent, replies["unknown"]))


def gen_extra(seed: int = 2026, per_intent: int = 120) -> List[Dict]:
    """Gen thêm nhiều mẫu với ngôn ngữ tự nhiên hơn."""
    random.seed(seed)
    examples: List[Dict] = []
    for intent, phrases in REAL_PHRASES.items():
        for _ in range(per_intent):
            level_code = random.choice(list(LEVELS.keys()))
            level_name = LEVELS[level_code][0]
            depth = LEVELS[level_code][1]
            phrase = random.choice(phrases)
            # Điền placeholder nếu có
            if "{depth_new}" in phrase:
                delta = random.randint(10, 40)
                phrase = phrase.replace("{depth_new}", str(depth + delta
                    if "cao hơn" in phrase or "Sâu hơn" in phrase else max(1, depth - 15)))
            if "{level_hint}" in phrase:
                phrase = phrase.replace("{level_hint}", level_name)
            examples.append(_make_example(phrase, intent, level_code, depth))
    # Mở đầu chat chưa có ảnh
    for t in NO_CONTEXT_REAL:
        examples.append(_make_example(t, "unknown", None, None))
    return examples


def export_real_chat() -> List[Dict]:
    """Xuất hội thoại thật từ AgentMemory thành training samples."""
    if not MEM_FILE.exists():
        return []
    try:
        with open(MEM_FILE, encoding="utf-8") as f:
            mem = json.load(f)
    except Exception:
        return []

    samples: List[Dict] = []
    history = mem.get("chat_history", [])
    # Gom thành từng cặp user→assistant
    for i, msg in enumerate(history):
        if msg.get("role") != "assistant":
            continue
        # Tìm user message đứng trước
        prev = None
        for j in range(i - 1, -1, -1):
            if history[j].get("role") == "user":
                prev = history[j]
                break
        if not prev:
            continue
        user_text = str(prev.get("content", ""))[:2000]
        asst_text = str(msg.get("content", ""))[:2000]
        if not user_text or not asst_text:
            continue
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": asst_text},
            ]
        })
    return samples


def save_dataset(examples: List[Dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Shuffle trước khi split để eval phân bố đều mọi intent/level
    rng = random.Random(2026)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.9)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in shuffled[:split]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for ex in shuffled[split:]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[Upgrade] Tổng: {len(shuffled)}")
    print(f"[Upgrade] Train: {split} | Eval: {len(shuffled) - split}")


def main() -> None:
    args = sys.argv[1:]
    do_gen = any(a in args for a in ("--gen", "--all"))
    do_export = any(a in args for a in ("--export", "--all"))
    if not do_gen and not do_export:
        print("Dùng: --gen (thêm mẫu tự nhiên) | --export (hội thoại thật) | --all")
        return

    # Backup tự động nếu chưa có
    backup_train = TRAIN_FILE.with_name(TRAIN_FILE.name + ".bak")
    backup_eval = EVAL_FILE.with_name(EVAL_FILE.name + ".bak")
    if not backup_train.exists():
        shutil.copyfile(TRAIN_FILE, backup_train)
        shutil.copyfile(EVAL_FILE, backup_eval)

    # Luôn gen từ BACKUP GỐC (data gốc) để kết quả tất định, tránh phình mỗi lần chạy
    base = _load_jsonl(backup_train) + _load_jsonl(backup_eval)
    print(f"[Upgrade] Data gốc (từ backup): {len(base)} mẫu")
    dataset = list(base)

    if do_gen:
        extra = gen_extra()
        print(f"[Upgrade] Gen thêm: {len(extra)} mẫu tự nhiên")
        dataset += extra
    if do_export:
        real = export_real_chat()
        print(f"[Upgrade] Hội thoại thật: {len(real)} mẫu")
        dataset += real

    save_dataset(dataset)


if __name__ == "__main__":
    main()
