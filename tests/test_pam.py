import subprocess
import shutil
import io
import sys
import os

class StreamBranch:
    def __init__(self, output1, output2):
        self.output2 = output2
        self.output1 = output1

    def write(self, text):
        self.output1.write(text)
        self.output2.write(text)

    def flush(self):
        self.output2.flush()

    def fileno(self):
        return self.output2.fileno()

expected_pam_metrics = """test/acc            0.6153846383094788
      test/f1_score         0.4380681812763214
       test/fbeta            0.418300598859787
        test/loss           1.3597944974899292
     test/precision         0.4089285731315613
       test/recall          0.4958333373069763"""

expected_ew_metrics = """test/acc            0.4166666567325592
       test/fbeta           0.26228001713752747
        test/loss           1.4233660697937012"""
# bewlow is the second try (data downloaded)

def run_test(config: str, expected: str):
    print(f"Running test for {config}")
    command = [
        "python",
        "/starformer/scripts/training/train.py",
        f"+experiment=final/{config}",
        "training.epochs=5"
    ]
    print("Training...")
    stdout = io.StringIO()
    stderr = io.StringIO()
    stdout_branch = StreamBranch(stdout, sys.stdout)
    stderr_branch = StreamBranch(stderr, sys.stderr)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    # stderr = stderr.read()
    # stdout = stdout.read()
    exit_code = process.returncode
    print(f"Stdout: {stdout}")
    if exit_code != 0:
        raise RuntimeError(f"Program exited with code {exit_code}. Stderr: {stderr}")
    else:
        assert expected in stdout, f"Test metrics were wrong."
        print("Tests passed.")

if __name__ == '__main__':
    print("testing with no data")
    if os.path.exists('/starformer/data'):
        shutil.rmtree('/starformer/data')
    run_test('pam-starformer.yaml', expected_pam_metrics)
    run_test('eigenworms-starformer.yaml', expected_ew_metrics)
    print("testing with data")
    run_test('pam-starformer.yaml', expected_pam_metrics)
    run_test('eigenworms-starformer.yaml', expected_ew_metrics)
