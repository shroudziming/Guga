from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guga.agent.model_adapter import AgentModelAdapter
from guga.agent.runner import AgentTaskRunner
from guga.benchmark.agent_tasks import run_agent_task_benchmark
from guga.config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, default_generation_config
from guga.memory.manager import MemoryManager
from guga.models import create_chat_model
from guga.persona import PersonaManager
from guga.tools import default_tool_registry
from guga.utils.paths import personas_dir


def main() -> None:
    args = _parse_args()
    model = create_chat_model(model_id=args.model_id, cache_dir=args.cache_dir)
    persona = PersonaManager(personas_dir()).load(args.persona)
    generation = default_generation_config()

    def runner_factory(case, workspace: Path, runs_root: Path) -> AgentTaskRunner:
        tools = default_tool_registry(workspace)
        memory_manager = MemoryManager(
            memory_root=runs_root.parent / "memory",
            model=model,
            enable_semantic=False,
            documents_dir=workspace / "documents",
        )
        adapter = AgentModelAdapter(
            model,
            generation,
            persona.system_prompt,
            tools,
        )
        return AgentTaskRunner(
            adapter,
            tools,
            memory_manager,
            agent_id="benchmark",
            runs_root=runs_root,
            expression_tags=persona.expression_tags,
        )

    run_dir = run_agent_task_benchmark(
        runner_factory,
        args.output_root,
        run_id=args.run_id,
    )
    print(f"benchmark results: {run_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the six-case Guga agent capability benchmark.")
    parser.add_argument("--model-id", default=os.environ.get("Guga_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--cache-dir", default=os.environ.get("Guga_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
    parser.add_argument("--persona", default=os.environ.get("Guga_PERSONA", "default"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmarks" / "agent_tasks",
    )
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
