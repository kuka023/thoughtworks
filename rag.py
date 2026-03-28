"""
RAG 引擎：加载 Excel 政策文件 → TF-IDF 向量化 → 检索相关条款
不需要下载任何模型，依赖：pandas, openpyxl, scikit-learn
"""
import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(BASE, "data", "policies")


class PolicyRAG:
    def __init__(self):
        self.chunks: list[dict] = []
        self.vectors = None
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",   # 字符级 n-gram，天然支持中文无需分词
            ngram_range=(2, 4),
            min_df=1,
            max_features=50000,
        )
        self.loaded_files: list[str] = []

    def load(self, policy_dir: str = POLICY_DIR) -> int:
        """加载目录下所有 Excel 文件，返回加载的条款数"""
        self.chunks = []
        self.vectors = None
        self.loaded_files = []

        if not os.path.exists(policy_dir):
            print(f"[RAG] 政策目录不存在: {policy_dir}")
            return 0

        for fname in sorted(os.listdir(policy_dir)):
            if fname.endswith((".xlsx", ".xls")) and not fname.startswith("~$"):
                fpath = os.path.join(policy_dir, fname)
                before = len(self.chunks)
                self._load_excel(fpath, fname)
                after = len(self.chunks)
                self.loaded_files.append({
                    "filename": fname,
                    "chunks": after - before
                })
                print(f"[RAG] 已加载 {fname}: {after - before} 条规则")

        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            self.vectors = self.vectorizer.fit_transform(texts)
            print(f"[RAG] 索引完成，共 {len(self.chunks)} 条政策规则")
        else:
            print("[RAG] 未找到任何政策规则，AI审核将不含来源标注")

        return len(self.chunks)

    def _load_excel(self, filepath: str, filename: str):
        try:
            xf = pd.ExcelFile(filepath)
            for sheet_name in xf.sheet_names:
                df = pd.read_excel(xf, sheet_name=sheet_name, dtype=str)
                df = df.fillna("")
                for idx, row in df.iterrows():
                    values = [str(v).strip() for v in row.values if str(v).strip()]
                    if not values:
                        continue
                    # 用于检索的拼接文本
                    text = " ".join([f"{col}{val}" for col, val in zip(df.columns, row.values)
                                     if str(val).strip()])
                    # 用于展示的格式化文本
                    display = " | ".join([f"{col}: {val}" for col, val in zip(df.columns, row.values)
                                          if str(val).strip()])
                    self.chunks.append({
                        "text": text,
                        "display": display,
                        "source": f"{filename} · {sheet_name} · 第{idx + 2}行",
                        "filename": filename,
                        "sheet": sheet_name,
                    })
        except Exception as e:
            print(f"[RAG] 加载失败 {filepath}: {e}")

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.05) -> list[dict]:
        """检索最相关的 k 条政策规则"""
        if not self.chunks or self.vectors is None:
            return []
        try:
            q_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.vectors)[0]
            top_idx = np.argsort(scores)[::-1]
            results = []
            seen_display = set()
            for i in top_idx:
                if scores[i] < threshold:
                    break
                chunk = self.chunks[i]
                if chunk["display"] in seen_display:
                    continue
                seen_display.add(chunk["display"])
                results.append({
                    "rule": chunk["display"],
                    "source": chunk["source"],
                    "score": round(float(scores[i]), 3),
                })
                if len(results) >= k:
                    break
            return results
        except Exception as e:
            print(f"[RAG] 检索失败: {e}")
            return []

    def retrieve_for_submission(self, submission: dict) -> list[dict]:
        """针对一份报销单，构建多维度查询并合并结果"""
        expense_types = list(set(item["type"] for item in submission.get("items", [])))
        city = submission.get("city", "")
        issues = submission.get("issues", [])

        # 多个查询维度
        queries = [
            f"{' '.join(expense_types)} {city} 报销标准 限额",
            "发票抬头 发票类型 合规要求",
            f"{'|'.join(issues) if issues else '报销规则'}",
        ]

        all_results: list[dict] = []
        seen = set()
        for q in queries:
            for r in self.retrieve(q, k=3):
                if r["source"] not in seen:
                    seen.add(r["source"])
                    all_results.append(r)

        # 按 score 排序，最多返回 6 条
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:6]

    def status(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "files": self.loaded_files,
            "ready": len(self.chunks) > 0,
        }


# 全局单例
rag = PolicyRAG()
