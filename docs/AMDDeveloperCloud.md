# AMD Developer Cloud: SSH and MIGraphX validation

Use an AMD GPU Droplet with ROCm development tools to evaluate the independent
`MIGRAPHX` backend. This guide is a setup procedure, not evidence of a passing
model matrix. Current evidence belongs in [Backend coverage](BackendCoverage.md#ev-migraphx).

## Provision and connect

Follow the [official AMD getting-started guide](https://www.amd.com/pt/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html).
Add your **public** SSH key during VM creation, select it for the VM, and obtain
the current public IP from the VM overview. The documented login is `root`.
The overview also provides Quick Start instructions and a Web Console.

On your local machine, substitute the current instance IP below:

```bash
export AMD_HOST='root@YOUR_INSTANCE_IP'
export AMD_KEY="$HOME/.ssh/id_ed25519"
ssh -o IdentitiesOnly=yes -i "$AMD_KEY" "$AMD_HOST"
```

The private key stays on your local machine; the server stores the matching
public key in the login user's `~/.ssh/authorized_keys`. `IdentitiesOnly=yes`
prevents an SSH agent with many keys from exhausting authentication attempts.
Verify the server fingerprint through a trusted channel on first connection.
If the local client reports a bad owner or permissions on a system SSH config,
have that config corrected. For a direct connection, `ssh -F /dev/null` can
bypass it, but also bypasses configured users, ports, proxies and identity files;
provide any required options explicitly.

For an existing instance, use the
[DigitalOcean instructions for adding a public key](https://docs.digitalocean.com/products/droplets/how-to/add-ssh-keys/to-existing-droplet/).
In a console logged in as the intended SSH user, run:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
printf '%s\n' 'YOUR_PUBLIC_KEY_LINE' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Replace `YOUR_PUBLIC_KEY_LINE` with the complete contents of your local `.pub`
file. For root login, these commands must run in root's home as root; adding a
key in another user's home authorizes that user instead. A non-root console user
needs appropriate administrative access to modify root's authorized keys.
Adding a key to the provider account alone does not update an existing VM.

## Diagnose connection failures

| Symptom | Check |
| --- | --- |
| `Permission denied (publickey)` | Confirm login user, selected private key, matching server public key, file ownership and permissions. |
| `Connection refused` | Confirm the current IP and port in the overview, VM readiness, SSH listener and firewall rejection rules. This happens before key authentication and does not identify the cause by itself. |
| Connection timeout | Check VM state, network route and provider/host firewall rules. |
| Host key changed | Verify the replacement instance's fingerprint before changing the saved host key. |

Use the Web Console when SSH is unavailable. On Ubuntu, inspect `systemctl status
ssh --no-pager` and `ss -ltnp`; starting the SSH service is appropriate only if it
is actually stopped. Keep diagnostic output private until reviewed.

## Inspect the runtime before building

On the instance:

```bash
rocm-smi
command -v hipcc
hipconfig --version
/opt/rocm/bin/rocm_agent_enumerator
cmake --version
c++ --version
```

Use the installed ROCm prefix if it differs. Record the detected GPU and `gfx...`
architecture rather than inferring versions or hardware from the image name.
Check access to `/dev/kfd` and `/dev/dri` if no GPU is visible. A successful build
does not establish GPU execution.

The companion [gpu-perf-lab cloud guide](https://github.com/olibartfast/gpu-perf-lab/blob/master/docs/setup/amd-developer-cloud.md)
covers HIP kernel builds and source transfer on the same provider. Its compiler
flags, presets and benchmark results belong to that repository; use this repo's
CMake entrypoints and neuriplo's MIGraphX dependency instructions here.

## Prepare and validate MIGraphX

Use [Dependency management](DependencyManagement.md) and the upstream
[MIGraphX setup documentation](https://github.com/olibartfast/neuriplo/blob/develop/docs/DEPENDENCY_MANAGEMENT.md#migraphx)
for SDK requirements. The independent backend consumes ONNX models; ONNX
Runtime's MIGraphX execution provider is a separate configuration.

Inspect existing checkouts with `git status` before updating them. For a
development campaign, use explicitly selected `develop` snapshots of
neuriplo-infer, neuriplo, neuriplo-tasks and videocapture. Record each commit and
any local changes, and confirm the configured build actually consumes those
snapshots. Normal builds use the release pins; simply pulling sibling repos
does not establish which sources CMake consumed. Preserve existing work by using
fresh checkout/build directories when necessary.

From the neuriplo-infer checkout, after installing the required dependencies:

```bash
cmake -S . -B build-mgx -DCMAKE_BUILD_TYPE=Release \
  -DDEFAULT_BACKEND=MIGRAPHX -DENABLE_APP_TESTS=ON
cmake --build build-mgx --parallel 8
ctest --test-dir build-mgx --output-on-failure
```

Run each command only after the previous one succeeds. Inspect configure output
and the CMake cache for the selected backend, SDK and dependency sources. Bound
build parallelism to available RAM. Run upstream MIGraphX reference and GPU
contract tests as described by the selected neuriplo revision before the app
matrix; retain failures and skips explicitly.

For learned-model cases, follow [Slurm backend coverage checks](SlurmBackendCoverage.md)
when a Slurm controller and workers are available. The cloud VM alone does not
provide that cluster. Stage commands, binaries and fixtures at the declared
absolute paths, use explicit time and memory limits, and declare all artifacts
for hashing. The harness's default address-space limit may need increasing for
GPU libraries and larger models.

Retain separate CPU-reference and GPU results, selected device, source/SDK
versions, exact commands, model and executable hashes, logs, timings and output
comparisons. Exit zero is `RUNTIME_PASS`; numerical or rendered task evidence is
needed to promote a task gate. Include required auxiliary inputs and assets for
cases such as YOLOWorld or LightGlue image pairs. A backend contract test or a
HIP benchmark does not validate those application paths.

## Keep private data out of Git and shared evidence

- Use placeholders for instance IPs, hostnames, account names, key comments and
  personal filesystem paths in committed examples. This guide includes no live
  endpoint or personal key material.
- Never upload private keys, tokens, credentials or credential-bearing clone
  URLs. Public keys are not secrets, but personal keys and fingerprints are
  unnecessary in published setup instructions.
- Keep raw SSH diagnostics, console output, source archives, manifests and run
  logs outside Git until reviewed. They may contain endpoints, usernames, paths,
  environment variables or credentials. Transfer only allowlisted source files;
  exclude `.ssh`, `.env`, credential stores and unrelated home files.
- Publish sanitized evidence that preserves reproducibility: backend and SDK
  versions, GPU model/architecture, commits, artifact hashes, commands with
  placeholders, exit codes and comparisons. Review images and model outputs for
  private content before sharing them. Keep raw evidence privately when needed.

Collect and review results before releasing the VM. Use the provider console to
stop or destroy the instance according to its billing policy; closing SSH does
not release it. Update [Backend coverage](BackendCoverage.md#ev-migraphx) only
after executing and reviewing the gates; this setup guide adds no new PASS claim.
