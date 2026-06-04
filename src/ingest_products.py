from __future__ import annotations

import re
from pathlib import Path

from llm import LLMClient
from retrieval import Retriever
from schema import ProductRecord, new_id


# ---------------------------------------------------------------------------
# 品牌名称 -> 品牌英文/简称 映射表（可自行扩展）
# ---------------------------------------------------------------------------
BRAND_MAP: dict[str, str] = {
    "可复美": "可复美",
    "珂岸": "珂岸",
    "科颜氏": "科颜氏",
    "左光": "左光",
    "珂润": "珂润",
    "谷雨": "谷雨",
    "芙芙": "芙芙",
}


def _detect_brand(name: str) -> str:
    """从产品名称前缀推断品牌"""
    for keyword, brand in BRAND_MAP.items():
        if name.startswith(keyword):
            return brand
    # fallback：取前两个字
    return name[:2] if len(name) >= 2 else name


def _extract_price_info(text: str) -> tuple[float | None, str | None, str]:
    """从价格块中提取价格、价格说明、净含量"""
    price_cny: float | None = None
    price_note: str | None = None
    net_content = ""

    # 价格行：类似 "246.05元人民币一盒"
    price_match = re.search(r"([\d.]+)\s*元", text)
    if price_match:
        price_cny = float(price_match.group(1))
        price_note = text.strip()

    # 净含量
    # 匹配模式：净含量50g、净含量为250ml、净含量：75g 等
    nc_match = re.search(r"净含量[：:为]?\s*(\d+\.?\d*\s*(?:g|ml|片|粒|mL|G|ML))", text, re.IGNORECASE)
    if nc_match:
        net_content = nc_match.group(1).strip()
    else:
        # 兜底：无"净含量"前缀的格式，如 "一盒5片面膜"、"一瓶30ml" 等
        fallback = re.search(r"(\d+\.?\d*\s*(?:g|ml|片|粒|mL|G|ML))\s*(?:面膜|瓶|盒|支|管)", text, re.IGNORECASE)
        if fallback:
            net_content = fallback.group(1).strip()

    return price_cny, price_note, net_content


# ---------------------------------------------------------------------------
# 章节标题 -> ProductRecord 字段名 映射
# ---------------------------------------------------------------------------
SECTION_FIELD_MAP: dict[str, str] = {
    "产品名称": "name",
    "价格": "_price_raw",
    "配方设计": "_ingredient_raw",
    "成分": "_ingredient_raw",
    "结构及组成": "_ingredient_raw",
    "类别": "model_type",
    "型号": "model_type",
    "品牌内部系列": "series",
    "功效": "efficacy",
    "适合肤质": "_skin_types_raw",
    "适用肤质": "_skin_types_raw",
    "用法用量": "_usage_raw",
    "使用方法": "usage_steps",
    "官方给出的常见问题": "faq",
    "产品官方Q&A": "faq",
    "官方Q&A": "faq",
    "官方介绍效果": "efficacy",
}

# 可能同时包含功效描述的章节
EFFICACY_SECTIONS = {"功效", "官方介绍效果"}


def _parse_product_block(block: str) -> dict:
    """
    解析单个产品文本块为字典。
    每个章节以 `# 标题` 开头，内容为下一章节之前的所有行。
    """
    # 按 "# 标题" 切分
    sections = re.split(r"\n(?=# )", block.strip())
    data: dict[str, str | list[str]] = {}

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # 提取标题
        header_match = re.match(r"^#\s+(.+?)(?:\n|$)", sec)
        if not header_match:
            continue
        header = header_match.group(1).strip()
        # 提取内容（去掉标题行）
        content = sec[header_match.end():].strip()

        field = SECTION_FIELD_MAP.get(header)
        if field is None:
            continue

        if field == "efficacy":
            # 可能多个功效来源，累积拼接
            existing = data.get("_efficacy_parts", [])
            existing.append(content)
            data["_efficacy_parts"] = existing
        elif field == "faq":
            # FAQ 用原始文本存储，后续结构化
            data["_faq_raw"] = content
        elif field == "_ingredient_raw":
            data["_ingredient_raw"] = content
        elif field == "_price_raw":
            data["_price_raw"] = content
        elif field == "_skin_types_raw":
            data["_skin_types_raw"] = content
        elif field == "_usage_raw":
            data["_usage_raw"] = content
        else:
            data[field] = content

    return data


def parse_raw_file(filepath: str | Path) -> list[dict]:
    """
    解析护肤品原始文本文件，返回结构化产品字典列表。
    
    文件格式：每个产品以 `# 产品名称` 开始，使用 `# 标题` 标注各章节。
    """
    filepath = Path(filepath)
    raw_text = filepath.read_text(encoding="utf-8")

    # 以 "# 产品名称" 分割产品块
    # 前面可能有无用内容，取从第一个产品名开始
    if "# 产品名称" not in raw_text:
        # 尝试 "# 产品名称" 但可能有全角/半角问题
        # 使用宽松匹配
        product_blocks = [raw_text]
    else:
        # 先找到第一个 "# 产品名称"
        idx = raw_text.index("# 产品名称")
        raw_text = raw_text[idx:]
        # 按 "\n# 产品名称" 或 "# 产品名称" 分割（保留分隔符前的内容）
        product_blocks = re.split(r"\n(?=# 产品名称)", raw_text)

    products: list[dict] = []
    for block in product_blocks:
        block = block.strip()
        if not block:
            continue
        parsed = _parse_product_block(block)
        if parsed.get("name"):
            products.append(parsed)

    return products


def _extract_ingredients(text: str) -> tuple[list[str], str]:
    """
    从成分文本中提取成分列表和排序文本。
    格式可能是：
      - "成分：水、甘油、..." 
      - 分类形式（保湿成分：... 舒缓成分：...）
    """
    ingredients: list[str] = []
    ordered_text = ""

    if not text:
        return ingredients, ordered_text

    # 移除子标题（如 "保湿成分："、"保湿成分:"）
    # 将所有成分行合并
    lines = text.split("\n")
    all_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 如果包含 "成分" 关键词，尝试提取冒号后的内容
        if "：" in line or ":" in line:
            # 分类行，提取冒号后内容
            parts = re.split(r"[：:]", line, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                all_parts.append(parts[1].strip())
        else:
            all_parts.append(line)

    combined = "、".join(all_parts)
    # 按分隔符拆分
    for sep in ["、", "，", ","]:
        combined = combined.replace(sep, "|")
    raw_ingredients = [s.strip() for s in combined.split("|") if s.strip()]

    # 去重并保留顺序
    seen: set[str] = set()
    ordered_list: list[str] = []
    for ing in raw_ingredients:
        ing_clean = ing.strip()
        if ing_clean and ing_clean not in seen and len(ing_clean) > 1:
            seen.add(ing_clean)
            ordered_list.append(ing_clean)

    return ordered_list, "、".join(ordered_list)


def _extract_skin_types(text: str) -> list[str]:
    """从适用肤质文本中提取肤质类型列表"""
    types: list[str] = []
    if not text:
        return types

    keywords = ["干性", "油性", "中性", "混合", "敏感", "任何肤质", "所有肤质", "痘肌"]
    for kw in keywords:
        if kw in text:
            types.append(kw)

    if "任何肤质" in text or "所有肤质" in text:
        return ["任何肤质"]

    return types if types else [text.strip()]


def _extract_concerns(text: str) -> list[str]:
    """从功效/描述文本中提取护肤关注点"""
    concern_keywords = [
        "保湿", "修护", "修复", "控油", "祛痘", "去痘", "美白", "祛斑",
        "淡斑", "抗皱", "紧致", "舒缓", "抗氧", "抗氧化", "提亮",
        "去角质", "清洁", "防晒", "淡痘印", "去闭口",
    ]
    concerns: list[str] = []
    for kw in concern_keywords:
        if kw in text:
            concerns.append(kw)
    return concerns


def _parse_faq(text: str) -> list[str]:
    """将 FAQ 文本解析为 Q&A 列表"""
    if not text:
        return []

    faqs: list[str] = []
    # 模式：问题行后跟答案行，问题以 ? 或 ？结尾
    lines = text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # 检测是否是问题行（以 ? 或 ？结尾，或包含"？"后的内容）
        if "？" in line or "?" in line:
            question = line
            answer_parts: list[str] = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue
                # 如果下一行是新问题，停止
                if "？" in next_line or "?" in next_line:
                    break
                answer_parts.append(next_line)
                i += 1
            answer = " ".join(answer_parts)
            faqs.append(f"Q: {question}\nA: {answer}")
        else:
            i += 1

    return faqs


def _extract_core_efficacy(efficacy_text: str) -> str:
    """从功效描述中提取核心功效"""
    if not efficacy_text:
        return ""
    # 模式："核心功效为XXX" 或 "核心功效是XXX"
    match = re.search(r"核心功效[为是][：:]?\s*(.+?)(?:[，,。\n]|$)", efficacy_text)
    if match:
        return match.group(1).strip()
    return ""


def build_product_records(parsed_products: list[dict], source_file: str) -> list[ProductRecord]:
    """
    将解析后的产品字典转换为 ProductRecord 对象列表。
    """
    records: list[ProductRecord] = []

    for p in parsed_products:
        name = p.get("name", "").strip() if isinstance(p.get("name"), str) else ""
        if not name:
            continue

        brand = _detect_brand(name)
        # 品牌也可能来自显式字段
        if p.get("brand"):
            brand = str(p["brand"])

        # 价格信息
        price_raw = str(p.get("_price_raw", ""))
        price_cny, price_note, net_content = _extract_price_info(price_raw)

        # 成分
        ingredient_raw = str(p.get("_ingredient_raw", ""))
        ingredients, ingredient_ordered_text = _extract_ingredients(ingredient_raw)

        # 肤质
        skin_raw = str(p.get("_skin_types_raw", ""))
        skin_types = _extract_skin_types(skin_raw)

        # 功效
        efficacy_parts: list[str] = p.get("_efficacy_parts", [])
        efficacy_text = "\n".join(efficacy_parts) if efficacy_parts else str(p.get("efficacy", ""))
        core_efficacy = _extract_core_efficacy(efficacy_text)

        # 关注点
        all_descriptive = f"{efficacy_text} {skin_raw} {name}"
        concerns = _extract_concerns(all_descriptive)

        # 使用说明
        usage_raw = str(p.get("_usage_raw", ""))
        usage_steps = str(p.get("usage_steps", ""))
        # usage_notes 合并 usage_raw 和 usage_steps
        usage_notes_parts = []
        if usage_raw and usage_raw != usage_steps:
            usage_notes_parts.append(usage_raw)
        if usage_steps:
            usage_notes_parts.append(usage_steps)
        usage_notes = "\n".join(usage_notes_parts)

        # FAQ
        faq_raw = str(p.get("_faq_raw", ""))
        faq = _parse_faq(faq_raw)

        # 型号/类别/系列
        model_type = str(p.get("model_type", ""))
        series = str(p.get("series", ""))

        # 按产品名称关键词推断分类（model_type 仅作补充，不覆盖分类）
        category = ""
        if "面膜" in name:
            category = "面膜"
        elif "敷料" in name:
            category = "医用敷料"
        elif "霜" in name:
            category = "面霜"
        elif "精华" in name:
            category = "精华"
        elif "水" in name or "爽肤" in name:
            category = "爽肤水"
        elif "乳" in name:
            category = "乳液"
        elif "洁颜" in name or "洁面" in name or "洁面泡" in name:
            category = "洁面"
        # model_type 作为 category 补充（仅在无法推断时使用）
        if not category and model_type:
            category = model_type

        # 贮存说明
        storage = ""
        if "贮存" in usage_raw:
            storage_match = re.search(r"贮存[：:]\s*(.+?)(?:\n|$)", usage_raw)
            if storage_match:
                storage = storage_match.group(1).strip()

        # 注意事项
        warnings = ""
        if "避开" in usage_steps or "不适" in usage_raw or "创面" in usage_raw:
            warnings = usage_raw

        record = ProductRecord(
            product_id=new_id(),
            name=name,
            brand=brand,
            category=category,
            price_cny=price_cny,
            price_note=price_note,
            ingredients=ingredients,
            ingredient_ordered_text=ingredient_ordered_text,
            skin_types=skin_types,
            concerns=concerns,
            usage_notes=usage_notes,
            source=source_file,
            efficacy=efficacy_text,
            core_efficacy=core_efficacy,
            faq=faq,
            model_type=model_type,
            series=series,
            net_content=net_content,
            storage=storage,
            usage_steps=usage_steps,
            warnings=warnings,
        )
        # 构建搜索文本
        record.search_text = record.build_search_text()
        records.append(record)

    return records


def ingest_products_to_qdrant(
    filepath: str | Path,
    embedder,  # LLMClient | LocalEmbedder，只要有 embed(texts) -> list[list[float]] 方法即可
    retriever: Retriever,
) -> list[ProductRecord]:
    """
    完整流程：解析原始文件 → 构建 ProductRecord → embedding → 存入 Qdrant。
    返回成功入库的 ProductRecord 列表。
    """
    filepath = Path(filepath)

    # 1. 解析原始文本
    parsed = parse_raw_file(filepath)

    # 2. 构建 ProductRecord
    records = build_product_records(parsed, source_file=str(filepath))

    # 3. 生成 embedding 并批量写入 Qdrant
    if records:
        search_texts = [r.search_text for r in records]
        vectors = embedder.embed(search_texts)

        for record, vector in zip(records, vectors):
            retriever.upsert_payload(
                collection=retriever.products_collection,
                point_id=record.product_id,
                vector=vector,
                payload=record.to_payload(),
            )

    return records
