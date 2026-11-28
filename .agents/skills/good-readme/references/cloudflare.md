# Cloudflare Ecosystem README Checklist

Additional checks for projects that target the Cloudflare platform. Apply these alongside the general [quality checklist](quality-checklist.md).

## Checklist

### Deploy to Cloudflare Button

Include a one-click deploy button — this is the strongest community signal for Cloudflare projects:

```markdown
[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=<REPO_URL>)
```

The button auto-provisions KV namespaces, D1 databases, R2 buckets, and other resources by reading the Wrangler config (`wrangler.jsonc` or `wrangler.toml`). Limitations: Workers apps only (not Pages), public repos on github.com and gitlab.com only.

### Free-Tier Compatibility

State whether the project runs entirely on the Cloudflare free tier. If paid features are required, list them explicitly so users know the cost before they start.

```markdown
## Requirements

Runs entirely on the Cloudflare free tier. No paid plans required.
```

Or, when paid features are needed:

```markdown
## Requirements

Requires a Workers Paid plan for Containers. All other services used are free tier.
```

Check [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/) for current plan requirements — these change over time.

### Cloudflare Primitives Declaration

Clearly state which Cloudflare services the project uses — don't make users guess. Use a compatibility table or bulleted list:

```markdown
## Cloudflare Services

| Service          | Purpose                        |
|------------------|--------------------------------|
| Workers          | API and request handling       |
| D1               | Serverless SQL database        |
| R2               | Object storage                 |
| KV               | Key-value data storage         |
| Durable Objects  | Stateful coordination          |
```

## Sources

- [Cloudflare Deploy Buttons](https://developers.cloudflare.com/workers/platform/deploy-buttons/)
- [Workers Pricing](https://developers.cloudflare.com/workers/platform/pricing/)
