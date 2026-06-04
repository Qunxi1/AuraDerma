"""护肤品数据入库脚本 —— 使用 Qdrant 本地存储 + 本地 Embedding 模型

解析原始文本文件，embedding 后存入本地 Qdrant 数据库。

用法:
    python run_ingest_products.py [文件路径]
    
默认读取 data/raw/护肤品介绍文档1.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目 src 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=True)

from qdrant_client import QdrantClient

from config import load_config
from llm import LocalEmbedder
from ingest_products import ingest_products_to_qdrant
from retrieval import Retriever


def main(filepath: str | None = None) -> None:
    if filepath is None:
        filepath = "data/raw/护肤品介绍文档1.jsonl"

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"错误: 文件不存在 - {filepath}")
        sys.exit(1)

    cfg = load_config()

    # 使用本地 Qdrant 存储（不需要服务器）
    qdrant_path = cfg.data_dir / "qdrant_local"
    qdrant_path.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] Qdrant 本地存储路径: {qdrant_path}")

    qdrant = QdrantClient(path=str(qdrant_path))

    # 加载本地 embedding 模型
    print(f"[2/5] 加载本地 Embedding 模型 (BAAI/bge-small-zh-v1.5)...")
    embedder = LocalEmbedder()
    print(f"      向量维度: {embedder.dim}")

    retriever = Retriever(
        qdrant,
        cfg.qdrant_collection_products,
        cfg.qdrant_collection_memory,
        cfg.qdrant_collection_docs,
    )
    print(f"[3/5] 初始化 collections (dim={embedder.dim})...")
    retriever.ensure_collections(vector_size=embedder.dim)

    print(f"[4/5] 解析产品数据: {filepath}")
    records = ingest_products_to_qdrant(filepath, embedder, retriever)

    print(f"\n{'='*50}")
    print(f"入库完成! 共处理 {len(records)} 个产品:")
    print(f"{'='*50}")
    for i, r in enumerate(records):
        print(f"\n{i+1}. {r.name}")
        print(f"   ID: {r.product_id}")
        print(f"   品牌: {r.brand}")
        print(f"   类别: {r.category}")
        if r.model_type:
            print(f"   型号: {r.model_type}")
        if r.series:
            print(f"   系列: {r.series}")
        if r.price_cny:
            print(f"   价格: {r.price_cny}元")
        print(f"   净含量: {r.net_content}")
        print(f"   肤质: {'; '.join(r.skin_types) if r.skin_types else '未标注'}")
        if r.efficacy:
            print(f"   功效: {r.efficacy[:80]}")
        if r.core_efficacy:
            print(f"   核心功效: {r.core_efficacy}")
        print(f"   关注点: {'; '.join(r.concerns)}")
        print(f"   成分数: {len(r.ingredients)}")
        print(f"   FAQ: {len(r.faq)} 条")

    # 验证：检索测试
    print(f"\n{'='*50}")
    print("[5/5] 检索验证: 搜索 '祛痘控油护肤品' ...")
    query_vec = embedder.embed(["祛痘控油护肤品推荐"])[0]
    results = retriever.search(retriever.products_collection, query_vec, limit=3)
    for hit in results:
        name = hit.payload.get("name", "?")
        brand = hit.payload.get("brand", "?")
        print(f"  score={hit.score:.3f} | [{brand}] {name}")

    print("\n全部完成!")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)
