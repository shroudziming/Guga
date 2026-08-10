from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "guga_cli.ps1"
TRACKED_CONFIG = PROJECT_ROOT / "config" / "guga_cli.env"


class GugaCliLauncherTest(unittest.TestCase):
    def test_tracked_config_defaults_are_explicit_and_non_sensitive(self) -> None:
        text = TRACKED_CONFIG.read_text(encoding="utf-8")

        self.assertIn("Guga_CLI_MODEL_ROUTE=api", text)
        self.assertIn("Guga_ENABLE_WRITE_TOOL=1", text)
        self.assertIn("Guga_ENABLE_COMMAND_TOOL=1", text)
        self.assertIn("1 = 允许", text)
        self.assertIn("0 = 禁止", text)
        self.assertNotIn("API_KEY=", text)
        self.assertNotIn("OPENAI_API_KEY=", text)

    def test_powershell_launcher_has_valid_syntax(self) -> None:
        escaped = str(LAUNCHER).replace("'", "''")
        command = (
            "$tokens=$null; $errors=$null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped}',"
            "[ref]$tokens,[ref]$errors) > $null; "
            "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
        )

        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_validate_only_loads_api_route_and_creates_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_file = root / ".env"
            config_file = root / "guga_cli.env"
            desktop = root / "Desktop"
            desktop.mkdir()
            env_file.write_text(
                "Guga_API_KEY=test-secret-value\n"
                "Guga_API_BASE_URL=https://example.invalid/v1\n",
                encoding="utf-8",
            )
            self._write_config(config_file, route="api")

            completed = self._run_validate(env_file, config_file, desktop)

            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertNotIn("test-secret-value", completed.stdout)
            payload = json.loads(completed.stdout)
            self.assertEqual("api", payload["model_provider"])
            self.assertEqual("api-test-model", payload["model_id"])
            self.assertEqual((desktop / "Guga").resolve(), Path(payload["workspace"]))
            self.assertTrue((desktop / "Guga").is_dir())
            self.assertEqual("1", payload["write_tool"])
            self.assertEqual("1", payload["command_tool"])

    def test_validate_only_selects_local_route_without_api_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_file = root / ".env"
            config_file = root / "guga_cli.env"
            desktop = root / "Desktop"
            desktop.mkdir()
            env_file.write_text("", encoding="utf-8")
            self._write_config(config_file, route="local")

            completed = self._run_validate(env_file, config_file, desktop)

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual("local", payload["model_provider"])
            self.assertEqual("local-test-model", payload["model_id"])
            self.assertEqual(
                (PROJECT_ROOT / "models_cache").resolve(),
                Path(payload["cache_dir"]),
            )

    @staticmethod
    def _write_config(path: Path, *, route: str) -> None:
        path.write_text(
            "\n".join(
                [
                    f"Guga_CLI_MODEL_ROUTE={route}",
                    "Guga_CLI_API_MODEL_ID=api-test-model",
                    "Guga_CLI_LOCAL_MODEL_ID=local-test-model",
                    "Guga_CLI_LOCAL_CACHE_DIR=./models_cache",
                    "Guga_CLI_DEFAULT_WORKSPACE=desktop",
                    "Guga_CLI_ALLOW_CREATE_WORKSPACE=1",
                    "Guga_ENABLE_WRITE_TOOL=1",
                    "Guga_ENABLE_COMMAND_TOOL=1",
                    "Guga_DEBUG=0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _run_validate(env_file: Path, config_file: Path, desktop: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LAUNCHER),
                "-ValidateOnly",
                "-EnvFile",
                str(env_file),
                "-ConfigFile",
                str(config_file),
                "-DesktopPath",
                str(desktop),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
