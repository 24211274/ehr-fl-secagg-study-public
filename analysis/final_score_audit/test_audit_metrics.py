"""Small transparent fixtures for the independent arithmetic used in audit_scores.py."""
import unittest
import numpy as np
from audit_scores import independent_ranking,counts
from sklearn.metrics import average_precision_score,roc_auc_score

class ArithmeticFixtures(unittest.TestCase):
    def close(self,r,ap,auc):
        self.assertAlmostEqual(r['AP'],ap,places=14);self.assertAlmostEqual(r['AUROC'],auc,places=14)
    def test_perfect(self):self.close(independent_ranking([0,0,1,1],[0,1,2,3]),1.,1.)
    def test_inverse_order(self):self.close(independent_ranking([1,1,0,0],[0,1,2,3]),5/12,0.)
    def test_all_tied(self):self.close(independent_ranking([1,0,0,1],[2,2,2,2]),.5,.5)
    def test_partial_ties(self):self.close(independent_ranking([1,0,1,0],[2,2,1,0]),7/12,5/8)
    def test_no_negative(self):self.assertEqual(independent_ranking([1,1],[2,3]),{'AP':None,'AUROC':None})
    def test_no_positive(self):self.assertEqual(independent_ranking([0,0],[2,3]),{'AP':None,'AUROC':None})
    def test_decision_denominators(self):
        r=counts([0,0,1,1],[1,0,1,0]);self.assertEqual([r[x] for x in ['TP','TN','FP','FN']],[1,1,1,1]);self.assertEqual(r['F1'],.5)
    def test_no_alert(self):
        r=counts([0,1],[0,0]);self.assertIsNone(r['precision']);self.assertEqual(r['F1'],0);self.assertEqual(r['FNR'],1)
    def test_affine_threshold_and_ranking(self):
        y=np.array([1,0,1,0]);x=np.array([.5,.5,.3,.1]);z=(x-.2)/.3
        self.assertEqual(independent_ranking(y,x),independent_ranking(y,z))
        self.assertTrue(np.array_equal(x>.3,z>((.3-.2)/.3)))
    def test_permutation_tied(self):
        y=np.array([1,0,1,0]);s=np.array([2,2,1,0]);ix=np.array([3,2,1,0])
        self.assertEqual(independent_ranking(y,s),independent_ranking(y[ix],s[ix]))
    def test_finite_input_guard(self):
        with self.assertRaises(AssertionError):independent_ranking([0,1],[0,np.nan])
    def test_deterministic_random_small_cases(self):
        g=np.random.default_rng(337)
        for n in (4,9,31,100):
            y=g.integers(0,2,n);y[:2]=[0,1];s=g.integers(0,8,n).astype(float)
            self.close(independent_ranking(y,s),average_precision_score(y,s),roc_auc_score(y,s))

if __name__=='__main__':unittest.main(verbosity=2)
