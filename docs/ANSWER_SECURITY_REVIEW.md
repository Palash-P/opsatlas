# Answer Path Security Review (Phase C)

> **Status note (Phase D, 2026-11-10):** Phase D added bearer-token identity,
> server-resolved tenants, tenant/ACL-scoped retrieval, citation metadata
> validation against server evidence, and redacted request telemetry. The
> authorization statements below describe the Phase C state and are superseded
> by `docs/ARCHITECTURE.md`, `docs/ENGINEERING_DECISIONS.md` (ADR-011–016), and
> `docs/PROJECT_STATUS.md`. The provider-error, prompt-injection, and
> version-confusion risks remain partly open as described there.

> Scope: `POST /api/v1/answer`, evidence assembly, grounded generation, citation
> validation, and the evaluation-only OpenCode provider. This is a focused
> threat-model review, not a penetration test.

## Trust boundaries and non-claims

The endpoint is **not public** and **authorization is not implemented**. A caller
supplies `tenant_id` in the request body and the service trusts it. Tenant
filtering exists in retrieval SQL, but it is not an access-control boundary until
identity-derived authorization replaces the request-body value.

The model provider is outside the trust boundary. Retrieved documents, model
output, and provider error text are all treated as untrusted input.

## Threats and controls

| Threat | Current control | Adversarial test | Residual risk |
| --- | --- | --- | --- |
| Cross-tenant retrieval | Retrieval receives exactly the request `tenant_id`; no fallback tenant is used. SQL filters candidates by tenant. | `test_answer_passes_request_tenant_to_retrieval`, `test_retrieval_dispatch_always_passes_the_request_tenant` | A caller can choose any tenant ID because authentication/authorization is absent. |
| Cross-tenant citation leakage | Citations are built only from the assembled bundle. A model-supplied ID that was not in the bundle is rejected. Empty retrieval short-circuits to abstention before the provider is called. | `test_cross_tenant_empty_retrieval_abstains_without_leaking`, `test_prompt_injection_cannot_authorize_unsupplied_citation` | If retrieval itself leaks a foreign chunk, the citation layer will faithfully cite it. Retrieval-level tenant isolation still needs dedicated authorization tests. |
| Prompt injection in retrieved documents | Evidence is labeled as untrusted data in both prompts. The model is told never to follow instructions inside evidence. | `test_prompt_injection_cannot_authorize_unsupplied_citation` | Instruction-following cannot be fully prevented by prompting. An injected passage may still bias a fluent, unsupported claim; faithfulness scoring is the detection control. |
| Evidence metadata forgery | `source_key`, `title`, `version`, `chunk_id`, and `chunk_index` are rendered as escaped attributes. Content is HTML-escaped so it cannot close or open an evidence block. API metadata comes from typed `EvidenceItem` fields, never from model output. | `test_forged_evidence_metadata_cannot_override_real_metadata`, `test_assembled_context_escapes_attribute_injection` | Escaping protects block boundaries, not semantic authority. A document can still *claim* to be another version in prose. Canonical metadata and the mixed-version indicator are the mitigation. |
| Version confusion | Version numbers come from database metadata, not document prose. Mixed versions are flagged per source. The prompt requires conflicts to name both evidence IDs. | `test_version_metadata_stays_canonical_when_content_claims_another_version` | The model can still attribute a procedure to the wrong version. That is measured by `conflict_handling` and the `version_conflict` subset. |
| Provider error leakage | The API maps provider, timeout, rate-limit, invalid-output, and configuration errors to fixed details. Exception text is not returned. The OpenAI-compatible adapter does not include response bodies or API keys in error messages. The evaluation adapter reports only the exit code, not raw stderr. | `test_provider_error_detail_does_not_leak_provider_message`, existing `tests/test_answer_api.py` error-mapping tests | Server logs and evaluation traces may still contain diagnostic text. Provider responses should not be logged verbatim. Omitting stderr also reduces diagnosability; record exit codes and retry counts instead. |
| Sensitive-data requests | The prompt forbids invention and the corpus has no credentials. Unanswerable sensitive records expect abstention. | Dataset records `answer-v1-sensitive-api-key`, `answer-v1-sensitive-customer-cards` (scored in the evaluation) | This is behavior under a synthetic corpus, not a data-loss-prevention control. |

## Adversarial test coverage

`tests/test_answer_security.py` adds ten test cases:

1. Request tenant is forwarded to retrieval.
2. A different tenant with empty retrieval abstains, returns no citations or
   evidence, and never calls the provider.
3. Provider exception text containing a fake key and tenant identifier is not
   exposed in the HTTP response.
4. Forged `<evidence>` tags and metadata inside chunk content cannot override
   the real API metadata.
5. Injected instructions cannot make an unsupplied `E99` citation valid.
6. Prose claiming a different version does not change canonical version
   metadata, mixed-version detection, or citation versions.
7. Attribute injection in `source_key` and `title` is escaped in the prompt.
8. Retrieval dispatch forwards the tenant for hybrid, lexical, and semantic
   strategies.

The existing suite already covers evidence-block boundary injection and
fabricated-citation rejection.

## Recommendations

1. Derive tenant and principal from authenticated identity; remove `tenant_id`
   from the request body or require it to match the authenticated principal.
2. Add database-level cross-tenant tests that seed two tenants and assert no
   candidate, excerpt, or citation crosses the boundary.
3. Treat retrieved content as data in any future tool-using provider. The current
   evaluation adapter refuses tool events rather than allowing workspace access.
4. Keep provider error details in server logs only, and redact request headers,
   API keys, and response bodies.
5. Add a conflict-specific schema (for example, required `versions` per conflict)
   if measured conflict-handling errors remain high.
