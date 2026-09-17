# Jev / TypeSafe constraint

Use the installed TypeSafe skill and Jev only where it genuinely fits, most naturally as an adaptive policy decision/classification surrogate. Do not force it into every component.

Hard boundaries:

- The latent ground-truth generator and core data-generating process remain fully deterministic and independent of Jev. Jev must never define latent truth, measurement truth, outcomes, or the criterion against which the method is judged.
- Retain a deterministic non-Jev adaptive policy arm so the central result is reproducible offline and any Jev-specific effect is separable from adaptivity.
- Treat Jev as an optional policy implementation or surrogate alongside the deterministic policy, never as the only policy.
- Log the requested model name, returned model/version, exact question instructions/criteria, bounded nonsecret state, typed outputs/probabilities, latency, HTTP status/failure class, and request/trace IDs only if the service returns them. Do not invent seeds or IDs Jev does not expose.
- The API credential stays in the vault. Never echo it or write it to disk, logs, env files, shell history, fixtures, results, or source. A real Jev run requires vault-mediated ephemeral injection and must be separately orchestrated; code/tests use a fake transport and sanitized recorded schema, not a credential.
- API failure must degrade to an explicitly labeled missing/failed Jev arm, never silently substitute outputs into the deterministic arm.
- No Jev output may change pre-frozen DGP parameters, certificate thresholds, or analysis choices.

The morning handoff must separate deterministic central results from optional Jev-arm results and state whether a live Jev grid ran. Do not spend API calls merely to decorate the build.
