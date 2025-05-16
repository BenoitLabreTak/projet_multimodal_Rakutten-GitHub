import subprocess

def run_pytest_test(test_file_path: str):
    result = subprocess.run(
        ["pytest", test_file_path, "--tb=short", "-q"],
        capture_output=True, text=True
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }