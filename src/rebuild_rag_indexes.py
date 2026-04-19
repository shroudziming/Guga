from __future__ import annotations

from guga.memory import MemoryManager


def main() -> None:
    manager = MemoryManager(debug=True)
    result = manager.rebuild_rag_indexes(session_id="manual_rebuild")
    print(
        "RAG 索引重建完成: "
        f"memory_chunks={result['memory_chunks']}, "
        f"document_chunks={result['document_chunks']}, "
        f"total_chunks={result['total_chunks']}"
    )


if __name__ == "__main__":
    main()
