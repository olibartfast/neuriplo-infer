# Model Pyramid — free models (opencode + 3rd-party free)

Living record for the agent model pyramid per skill
`select-opencode-free-models`, extended to 3rd-party free endpoints.
Update it on the cadence below; CI skips `docs/**` so doc-only refreshes are cheap.

- Ultimo aggiornamento: 2026-11-17 (EU/IT)
- Arena open-source check: 2026-11-14 (snapshot skill: DeepSeek top-5, GLM top-10, MiniMax top-15, Nemotron-SVL top-20, Lightning top-25; Hy4 / north-mini-code unranked)
- Docs OpenCode: https://opencode.ai/docs/console/models/ (sezione Free models + Privacy + Deprecated)
- Arena: https://arena.ai/leaderboard/agent?license=open-source (filtro open-source)

## Piramide attiva

| Ruolo | Modello | Variant | Privacy | Rate limit noto | Fallback |
|---|---|---|---|---|---|
| Architect | `opencode/deepseek-v4.1-flash-free` | `max` | zero-retention | nessun cap pubblicato | `opencode/glm-5.3-flash-free#max` |
| Orchestrator | `opencode/glm-5.3-flash-free` | `high` | zero-retention | nessun cap pubblicato | `opencode/deepseek-v4.1-flash-free#high` |
| Implementer | `opencode/deepseek-v4.1-flash-free` | `max` | zero-retention | nessun cap pubblicato | `opencode/glm-5.3-flash-free#max` |
| Reviewer | `opencode/glm-5.3-flash-free` | `max` | zero-retention | nessun cap pubblicato | `nvidia/minimaxai/minimax-m3.1-flash` bare — 3a famiglia, solo non-confidenziale, ~40 RPM shared |
| Explorer | `openrouter/cohere/north-mini-code:free` | — | sconosciuta/may-improve | 20 RPM, 50/giorno (<10$ lifetime) / 1000/giorno (≥10$) | `opencode/glm-5.3-flash-free#low` |
| Vision/velocità | `opencode/nemotron-3.5-super-vl-free` | `xhigh` | trial logged (waiver) | nessun cap pubblicato | `nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b` bare |

Scorte zero-retention: `opencode/hy4-flash-free#high`. `longcat-2.5-free` solo ultima risorsa (100/giorno, hosted Cina).

## Esclusi il 2026-11-17

- `amd/*`: `Invalid bearer token` (serve key provider)
- `nvidia/deepseek-ai/deepseek-v4.1-flash`: EOL 2026-11-16
- `nvidia/z-ai/glm-5.3-flash`: timeout
- `opencode/inkling-2-small-free`: geo-block EU/IT
- `opencode/minimax-m3.1-flash-free`, `opencode/nemotron-3-ultra-free`: unavailable/deprecato 2026-11-17

## Cadence di aggiornamento SOTA

Settimanale (Arena aggiorna i ranking ~ogni settimana) + event-driven:

1. `bash scripts/check_model_pyramid.sh` — catalogo + probe candidati piramide.
2. Rileggi sezione Free models / Privacy / Deprecated di `https://opencode.ai/docs/console/models/`.
3. Apri `https://arena.ai/leaderboard/agent?license=open-source`, registra rank e data per DeepSeek, GLM, MiniMax, Nemotron, nuovi entranti free.
4. Se cambia reachability, deprecation, privacy o Arena in modo materiale: aggiorna tabella sopra, probe-date e Arena-date, fallback stessa classe privacy (il waiver trial-logged resta esplicito, non implicito per nuovi modelli).
5. Trigger immediati (fuori cadence): `Model unavailable`, geo-block, 429/quota, docs/deprecation change, modello ritirato, cambio variant, shift Arena significativo.

## Verifica rapida

```sh
bash scripts/check_model_pyramid.sh
curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" | jq .data.free_model_daily_requests
# dashboard personale nvidia: build.nvidia.com (top-right) per il tuo RPM
```
