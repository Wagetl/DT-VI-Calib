import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "time_delay_experiments", Path(__file__).resolve().parents[1] / "scripts/time_delay_experiments.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ParseTraceTest(unittest.TestCase):
    def test_accepted_and_rejected_trials_keep_their_status(self):
        log = """[LINEARIZE] Error: 100 num points 50
setup_time 2.000000 ms.
\x1b[31mcam_time_offset 0.020000000 s.\x1b[0m
[REJECTED] lambda:1 step_quality: -1 max_inc: 1 Error: 110 num points 50
cam_time_offset 0.010100000 s.
[ACCEPTED] lambda:2 step_quality: 1 max_inc: 0.1 Error: 10 num points 50
iter_time 3.000000 ms.
Current error: 10 mean reprojection 0.08
Optimization Converged !!
total opt_time 6.000000 ms.
"""
        rows, total, reprojection = experiment.parse_log(log)
        self.assertEqual([row["status"] for row in rows], ["initial", "rejected", "accepted"])
        self.assertAlmostEqual(rows[-1]["estimate_ms"], 10.1)
        self.assertEqual(rows[-1]["iteration"], 1)
        self.assertEqual(rows[-1]["optimizer_elapsed_ms"], 5)
        self.assertEqual(total, 6)
        self.assertEqual(reprojection, 0.08)
        with self.assertRaises(ValueError):
            experiment.parse_log(log.replace("Optimization Converged !!", ""))
        with self.assertRaises(ValueError):
            experiment.parse_log(log.replace("cam_time_offset 0.010100000 s.", ""))


if __name__ == "__main__":
    unittest.main()
