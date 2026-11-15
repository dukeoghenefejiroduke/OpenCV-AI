"""Unit tests for Phase C: Evidence-Gated Classifier & Features."""

import unittest
import os
import tempfile
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.features import extract_features
from vision.logistic_classifier import NumpyLogisticRegression, wilson_lower_bound, check_evidence_gating


class TestPhaseC(unittest.TestCase):

    def test_feature_extraction(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feats = extract_features(img)
        self.assertIn("v_mean", feats)
        self.assertIn("roughness", feats)
        self.assertIn("edge_density", feats)
        self.assertIsInstance(feats["v_mean"], float)

    def test_logistic_regression_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(float)
        
        clf = NumpyLogisticRegression(epochs=100)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        self.assertEqual(len(probs), 50)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_wilson_lower_bound(self):
        lb = wilson_lower_bound(95, 100)
        self.assertGreaterEqual(lb, 0.0)
        self.assertLessEqual(lb, 1.0)

    def test_evidence_gating_refusal(self):
        # By default, validation_targets.json is unsealed, so check_evidence_gating should return False
        self.assertFalse(check_evidence_gating("data/validation_targets.json", {"wilson_lower_bound": 0.95}))
