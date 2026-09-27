"""Study 3 toy counterexample: observed randomized contrasts cannot label a target-channel effect as artifact.

Independent synthetic method probe only. No participant data, repo dependency, or source access.
Run: python3 identification_probe.py; tests: python3 -m unittest identification_probe.py
"""
from __future__ import annotations
import unittest
import numpy as np

# Y is three observed weekly outcomes [mood, steps, sleep].
# Delta Y(k) = lambda * tau(k) + direct(k). A non-target anchor restriction
# sets two entries of direct(k) to zero; it is an assumption, not something
# identified by randomization alone.
LOADING = np.array([1.0, 0.7, 0.5])
CATEGORIES = ('mood','steps','sleep')


def observed_effect(shared_latent_effect: float, direct_effect, loading=LOADING):
    return np.asarray(loading)*shared_latent_effect + np.asarray(direct_effect)


def anchor_decomposition(delta, target: int, loading=LOADING):
    """Return an *assumption-dependent* decomposition, not an identified causal latent effect.

    It requires a fixed loading scale, nonzero invariant anchor loadings, and no direct
    category effects on the non-target anchors. Agreement of the two anchor ratios
    is a necessary rejection diagnostic, never proof these exclusions hold.
    """
    anchors = [j for j in range(3) if j!=target]
    if any(abs(loading[j]) < 1e-9 for j in anchors):
        raise ValueError('zero-loading anchor: not identified')
    implied = np.asarray(delta)[anchors]/np.asarray(loading)[anchors]
    return float(implied.mean()),float(np.asarray(delta)[target]-loading[target]*implied.mean()),float(implied[0]-implied[1])


def run():
    # Algebraic counterexample: no participant or allocation data are simulated.
    # A true shared latent effect 0.2 with no direct effects:
    d=observed_effect(.2,[0,0,0]); shared=anchor_decomposition(d,0)
    # Observationally identical assigned-arm contrasts under an alternative world:
    # no shared latent effect, direct effects on *all* three observed streams.
    other=observed_effect(0,d)
    # This is exact observational equivalence even with unlimited randomized sample.
    assert np.allclose(d,other)
    # A category effect on steps of 0.20 can be real physical activity improvement
    # without improving the chosen shared mental-state construct.
    steps=observed_effect(0,[0,.2,0]); step_analysis=anchor_decomposition(steps,1)
    return dict(probe="algebraic synthetic contrast, not a trial simulation",
                observed_shared_contrasts=d.tolist(),identical_spillover_contrasts=other.tolist(),
                assumption_dependent_anchor_shared_effect=shared[0],assumption_dependent_anchor_direct_effect=shared[1],
                steps_only_contrasts=steps.tolist(),steps_target_effect=step_analysis[1])


class ProbeTests(unittest.TestCase):
    def test_shared_versus_spillover_observationally_equivalent(self):
        d=observed_effect(.2,[0,0,0]); alt=observed_effect(0,d)
        np.testing.assert_allclose(d,alt)
        latent,direct,disagreement=anchor_decomposition(d,0)
        self.assertAlmostEqual(latent,.2)
        self.assertAlmostEqual(direct,0.)
        self.assertAlmostEqual(disagreement,0.) # passes diagnostic despite false exclusion in alt world

    def test_targeted_behavior_is_not_necessarily_artifact(self):
        latent,direct,disagreement=anchor_decomposition([0,.2,0],1)
        self.assertAlmostEqual(latent,0.)
        self.assertAlmostEqual(direct,.2)
        self.assertAlmostEqual(disagreement,0.)

    def test_anchor_disagreement_is_rejection_not_identification(self):
        latent,direct,disagreement=anchor_decomposition([.1,.2,0],1)
        self.assertNotAlmostEqual(disagreement,0.)
        with self.assertRaisesRegex(ValueError,'not identified'):
            anchor_decomposition([.1,.2,0],1,loading=[1,.7,0])


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
