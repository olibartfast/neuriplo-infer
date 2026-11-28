# README Quality Checklist & Scoring Rubric

Use this checklist to audit a GitHub README and produce a quality score out of 100 points.

---

## Rating Scale

| Score     | Rating      | Meaning                                          |
|-----------|-------------|--------------------------------------------------|
| 90 - 100  | Exemplary   | Could be used as a template for other projects   |
| 75 - 89   | Strong      | Minor improvements possible                      |
| 60 - 74   | Adequate    | Gets the job done but has noticeable gaps         |
| 40 - 59   | Weak        | Missing important information                    |
| 0 - 39    | Poor        | Needs significant rework                         |

---

## Category 1: First Impressions (0-20 points)

The opening of a README determines whether a visitor stays or leaves. Score each criterion on 0-5.

### 1.1 Ten-Second Clarity (0-5)

Can a reader understand what this project does within 10 seconds of opening the README?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | The first 1-3 sentences plainly state what the project is and what problem it solves. A newcomer with no prior context can grasp the purpose immediately. |
| **Bad (0-1)** | The README opens with badges, build status, or a logo with no explanatory text. The reader must scroll or read multiple paragraphs before understanding the project's purpose. |

**How to check:** Read only the content above the first `##` heading (or the first ~5 lines). Can you answer "What does this project do?" in one sentence? If not, score low.

### 1.2 Clear Value Proposition (0-5)

Does the README explain *why* someone should use this project?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | Explicitly states the benefit — e.g., "10x faster than X", "eliminates the need to manually configure Y", or a concise feature list immediately after the description. |
| **Bad (0-1)** | No indication of why this project exists or what advantage it offers over alternatives. The reader cannot tell why they would choose this over doing nothing or using something else. |

**How to check:** Look for phrases that communicate benefit, comparison, or motivation in the first section. Check for a "Features" or "Why use this?" section near the top.

### 1.3 Descriptive Title (0-5)

Is the title (H1 or repo name) descriptive enough to convey meaning on its own?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | The title is a recognizable name accompanied by a subtitle or tagline that describes the project — e.g., "FastAPI — modern, fast web framework for building APIs with Python". |
| **Bad (0-1)** | The title is an acronym, codename, or single word with no subtitle, and the reader cannot infer the project's domain or purpose from the title alone. |

**How to check:** Read only the H1 heading. Does it communicate the project's domain? If it is a creative/brand name, is there a tagline within the same visual block?

### 1.4 Jargon-Free Opening (0-5)

Does the opening avoid unexplained jargon, acronyms, or domain-specific terms?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | The first paragraph uses plain language. Domain terms that appear are either self-evident from context or briefly defined. Accessible to a developer unfamiliar with the specific niche. |
| **Bad (0-1)** | The opening is dense with unexplained acronyms, framework-specific terms, or assumes the reader already knows the problem space. Only an insider would understand the first paragraph. |

**How to check:** Identify every technical term or acronym in the first paragraph. For each one, ask: would a general software developer understand this without extra context? Count unexplained terms — more than two is a red flag.

---

## Category 2: Getting Started (0-20 points)

A README that cannot get a user from zero to running is failing its primary job. Score each criterion on 0-5.

### 2.1 Complete Installation Steps (0-5)

Are installation steps complete and correct?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | Installation is a numbered or clearly sequenced set of commands/steps that cover the full path from "I have nothing" to "the software is installed." No implicit steps are omitted. |
| **Bad (0-1)** | Installation is missing, consists of a single `npm install` with no context, skips essential steps (e.g., cloning, environment setup), or references external docs without summarizing the key steps. |

**How to check:** Walk through the installation instructions mentally (or literally). Is every command provided? Does it say where to run them? Are there gaps where the reader would need to guess or search externally?

### 2.2 Quick-Start That Works (0-5)

Is there a quick-start or "Hello World" that works out of the box?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | After installation, the README shows a minimal working example — run one command or paste a few lines of code and see a result. The quick-start is clearly labeled and separated from advanced usage. |
| **Bad (0-1)** | No quick-start exists. The reader finishes installing and has no idea what to do next. Or the quick-start requires extensive configuration before producing any output. |

**How to check:** Look for a section labeled "Quick Start", "Getting Started", "Hello World", or "Basic Usage". Check whether it produces a visible result with minimal setup. Verify the commands/code reference the correct package name and API.

### 2.3 Prerequisites Listed (0-5)

Are prerequisites (language runtime, OS packages, tools) explicitly listed?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | Prerequisites are listed in a dedicated section or clearly stated before installation — e.g., "Requires Node.js >= 18, Docker, and a PostgreSQL database." |
| **Bad (0-1)** | Prerequisites are not mentioned. The user discovers missing dependencies only when a command fails. Or prerequisites are buried deep in the document. |

**How to check:** Search for "prerequisites", "requirements", "dependencies", or "before you begin." Check whether the installation section references tools without first confirming the reader has them.

### 2.4 Versions & Platforms (0-5)

Does the README specify which versions and platforms are supported?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | States supported OS (Linux, macOS, Windows), language/runtime versions, and any platform constraints. May include a compatibility matrix or badges. |
| **Bad (0-1)** | No mention of supported platforms or versions. The reader cannot tell if the project works on their system without trying it. |

**How to check:** Search for version numbers, OS names, "supported", "compatibility", or "tested on." Check whether the package.json / pyproject.toml / Cargo.toml version constraints are reflected in the README.

---

## Category 3: Usage & Examples (0-20 points)

Examples are the most-read section of any README. Score each criterion on 0-5.

### 3.1 Real, Working Code Examples (0-5)

Are there real, working code examples (not pseudocode)?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | Code examples use actual API calls, real function names, and syntactically valid code. They import the correct modules and use current method signatures. |
| **Bad (0-1)** | Examples are pseudocode, use placeholder names like `doSomething()`, or reference APIs that no longer exist. Or there are no code examples at all. |

**How to check:** Compare function names, import paths, and method signatures in the examples against the actual source code. Check that the language syntax is valid. Look at the project's exports/public API and verify the examples use them correctly.

### 3.2 Progressive Complexity (0-5)

Do examples progress from simple to complex?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | Examples start with the simplest possible use case and build toward advanced scenarios. A clear learning path exists: basic usage, then configuration options, then advanced/edge cases. |
| **Bad (0-1)** | The only example is a complex, real-world scenario with many options and parameters. Or examples are randomly ordered with no progression. A newcomer would not know where to start. |

**How to check:** Count the examples and assess their ordering. Does the first example have the fewest lines and fewest parameters? Does complexity increase as you scroll down?

### 3.3 Copy-Paste Ready (0-5)

Can examples be copy-pasted and run with minimal modification?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | Examples include all necessary imports, variable declarations, and context. A user can paste them into a file or REPL and execute them immediately. Placeholder values (API keys, file paths) are clearly marked. |
| **Bad (0-1)** | Examples are code fragments that omit imports, assume variables from prior context, or use unexplained placeholder values. The user must assemble multiple snippets to get something runnable. |

**How to check:** For each code block, ask: if I paste this into an empty file and run it (with the package installed), will it execute? Are all imports included? Are placeholder values like `YOUR_API_KEY` clearly indicated?

### 3.4 Expected Output Shown (0-5)

Is the expected output or result shown for examples?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | Each example is followed by its expected output, a screenshot, or a description of the result. The reader can verify they got the right result without guessing. |
| **Bad (0-1)** | No output is shown. The reader runs the example and has no way to know if the result is correct. |

**How to check:** Look for output blocks (text, JSON, screenshots, GIFs) immediately following code examples. Check whether the output matches what the code would actually produce.

---

## Category 4: Structure & Scannability (0-15 points)

Most readers scan; they do not read linearly. Score criteria as indicated.

### 4.1 Well-Organized Headers (0-4)

Is the README well-organized with clear, descriptive headers?

| Score | Description |
|-------|-------------|
| **Good (3-4)** | Headers form a logical hierarchy (H1 > H2 > H3). Section names are standard and predictable ("Installation", "Usage", "API", "Contributing"). A table of contents is present for long READMEs. |
| **Bad (0-1)** | Few or no headers. Sections are not logically ordered (e.g., "Contributing" appears before "Installation"). Header names are vague ("Stuff", "More Info") or inconsistent in style. |

**How to check:** Extract all headings. Do they form a logical outline? Is the order intuitive? For READMEs longer than ~200 lines, is there a table of contents?

### 4.2 Key Info Findable Without Full Read (0-4)

Can a reader find key information (install, usage, API) without reading the entire document?

| Score | Description |
|-------|-------------|
| **Good (3-4)** | Standard section names, a table of contents, or anchor links let the reader jump to what they need. Critical information (install command, primary import) appears early. |
| **Bad (0-1)** | Important details are buried in prose paragraphs. There is no way to jump to a specific topic. The reader must scan the entire document to find basic information. |

**How to check:** Try to find the install command, the primary usage pattern, and the license without reading sequentially. Time how long it takes conceptually — if it requires scanning more than two screens, score low.

### 4.3 Appropriate Length (0-4)

Is the length appropriate for the project's complexity?

| Score | Description |
|-------|-------------|
| **Good (3-4)** | The README is proportional to the project. A simple CLI tool has a concise README; a complex framework has a thorough one. Advanced details are linked out to separate docs rather than bloating the README. |
| **Bad (0-1)** | The README is either a single paragraph for a complex project (too short) or a 2000-line wall of text for a simple utility (too long). No external documentation is referenced when it should be. |

**How to check:** Estimate the project's complexity from its source. Compare the README length. A simple utility should be 50-200 lines; a framework or platform might justify 300-800 lines. Beyond 1000 lines, check whether content should be moved to a docs site.

### 4.4 Effective Formatting (0-3)

Are code blocks, lists, tables, and other Markdown formatting used effectively?

| Score | Description |
|-------|-------------|
| **Good (2-3)** | Code uses fenced blocks with language identifiers. Lists break up dense information. Tables present comparisons or option descriptions. Bold/italic highlight key terms sparingly. |
| **Bad (0)** | Code is inline or unformatted. Long paragraphs contain information that should be in lists or tables. No visual hierarchy beyond headers. |

**How to check:** Scan for fenced code blocks (with language tags), bullet/numbered lists, and tables. Check that inline code is used for commands, file names, and variable names. Verify that formatting aids comprehension rather than being decorative.

---

## Category 5: Accuracy & Maintenance (0-15 points)

A beautiful README that is wrong is worse than an ugly one that is right. Score criteria as indicated.

### 5.1 APIs/Commands Match Current Code (0-5)

Do the documented APIs, CLI commands, and function signatures match the current codebase?

| Score | Description |
|-------|-------------|
| **Good (4-5)** | Every function, method, CLI flag, and configuration option mentioned in the README exists in the current code with the same name and signature. |
| **Bad (0-1)** | The README references renamed functions, removed CLI flags, or deprecated APIs. The code has diverged significantly from the documentation. |

**How to check:** For each API call, function name, or CLI command in the README, search the codebase for a matching definition. Check that parameter names and types match. Pay special attention to recently changed files (check git log for the README vs. source files).

**API drift evidence standard:** Do not accept a symbol because it appears in an old README example. Confirm it is part of the current public surface: package manifest exports, `bin` entry, `__all__`, index/barrel export, generated types, CLI parser, or documented config schema. For each mismatch, record the stale README symbol, current source-backed symbol, evidence file, and whether the old name is removed, renamed, moved, or ambiguous.

### 5.2 Current Version Numbers & Dependencies (0-4)

Are version numbers, dependency names, and compatibility claims up to date?

| Score | Description |
|-------|-------------|
| **Good (3-4)** | Version numbers in the README match the latest release. Dependency versions align with the lockfile or manifest. Compatibility claims have been verified recently. |
| **Bad (0-1)** | The README references an old version number, lists dependencies that have been removed, or claims compatibility with versions that are no longer tested. |

**How to check:** Compare version numbers in the README against package.json / Cargo.toml / pyproject.toml / go.mod and the latest git tag or release. Check that install commands reference the correct package name and version.

### 5.3 Working Links (0-3)

Are all links functional (no 404s)?

| Score | Description |
|-------|-------------|
| **Good (2-3)** | All links resolve. Internal links point to existing files and headings. External links reach live pages. |
| **Bad (0)** | Multiple broken links. Links point to moved or deleted pages, renamed branches (e.g., `master` when the default is now `main`), or anchors that no longer exist. |

**How to check:** Extract all URLs and relative links from the README. For internal links, verify the target file or heading exists in the repo. For external links, check if they are to known-stable destinations (official docs, Wikipedia) or potentially ephemeral ones (blog posts, specific GitHub tree URLs with commit hashes).

### 5.4 Consistency with Other Docs (0-3)

Is the README consistent with other documentation in the project (CONTRIBUTING.md, docs/, wiki)?

| Score | Description |
|-------|-------------|
| **Good (2-3)** | The README aligns with other docs. It does not contradict the CONTRIBUTING guide, API docs, or changelog. Cross-references are accurate. |
| **Bad (0)** | The README contradicts other project documentation. For example, the README says "run `make build`" but the CONTRIBUTING guide says "run `cargo build`". |

**How to check:** If the repo has CONTRIBUTING.md, docs/, or a wiki, spot-check that key instructions (build, test, install) are consistent. Check that the README links to the correct docs and that linked docs agree with README content.

---

## Category 6: Completeness (0-10 points)

These items round out a professional README. Score criteria as indicated.

### 6.1 License Stated (0-3)

Is the license clearly identified?

| Score | Description |
|-------|-------------|
| **Good (2-3)** | The license is stated in the README (a "License" section or badge) and a LICENSE file exists in the repo root. The stated license matches the LICENSE file. |
| **Bad (0)** | No license is mentioned in the README and no LICENSE file exists. Or the README says one license but the LICENSE file says another. |

**How to check:** Search the README for "license" or "License". Check for a LICENSE or LICENSE.md file in the repo root. Verify they agree.

### 6.2 Contribution Guidance (0-3)

Is there guidance for contributors (if the project accepts contributions)?

| Score | Description |
|-------|-------------|
| **Good (2-3)** | A "Contributing" section exists with clear instructions or a link to CONTRIBUTING.md. It covers how to submit issues, pull requests, and any code style or testing requirements. |
| **Bad (0)** | No contributing guidance exists despite the project being open to contributions. Or the guidance is a single sentence like "PRs welcome" with no further detail. |

**How to check:** Look for a "Contributing" section in the README or a CONTRIBUTING.md file. If the project has open issues and merged PRs from external contributors, contribution guidance should exist. Score N/A (full marks) for personal/internal projects that are not seeking contributions.

### 6.3 Contact & Support Channels (0-2)

Are there contact or support channels listed?

| Score | Description |
|-------|-------------|
| **Good (2)** | The README lists where to get help — issue tracker, Discord, mailing list, Stack Overflow tag, or a "Support" section. Users know where to ask questions. |
| **Bad (0)** | No indication of where to report bugs, ask questions, or get support. |

**How to check:** Search for "support", "help", "community", "Discord", "Slack", "issues", "discussions", or "contact" in the README.

### 6.4 Limitations & Non-Goals (0-2)

Does the README cover what the project does NOT do or its known limitations?

| Score | Description |
|-------|-------------|
| **Good (2)** | A "Limitations", "Non-Goals", "Known Issues", or "Caveats" section exists. The reader is not surprised by missing functionality after adopting the project. |
| **Bad (0)** | No mention of limitations. The project appears to do everything, which is rarely true. Users discover limitations only after investing time. |

**How to check:** Search for "limitation", "caveat", "non-goal", "known issue", "not supported", or "out of scope" in the README. Also check whether the project description makes claims that could mislead (e.g., "full-featured" when major features are missing).

---

## Audit Procedure

When auditing a README, follow these steps:

1. **Read the README fully** before scoring anything.
2. **Score each of the 22 criteria** individually using the rubrics above.
3. **Sum the scores** across all six categories to get the total (out of 100).
4. **Map to the rating scale** (Exemplary / Strong / Adequate / Weak / Poor).
5. **List the top 3 strengths** — criteria that scored highest.
6. **List the top 3 improvements** — criteria that scored lowest, ordered by impact.
7. **Provide specific, actionable suggestions** for each improvement, with example text or structure where helpful.

### Output Format

```
## README Quality Audit

**Project:** [name]
**Score:** [X] / 100
**Rating:** [Exemplary | Strong | Adequate | Weak | Poor]

### Scores by Category

| Category                    | Score | Max |
|-----------------------------|-------|-----|
| First Impressions           |       | 20  |
| Getting Started             |       | 20  |
| Usage & Examples            |       | 20  |
| Structure & Scannability    |       | 15  |
| Accuracy & Maintenance      |       | 15  |
| Completeness                |       | 10  |
| **Total**                   |       | 100 |

### Detailed Scores

[List each of the 22 criteria with its score and a one-line justification]

### Strengths

1. [Criterion] — [Why it scored well]
2. [Criterion] — [Why it scored well]
3. [Criterion] — [Why it scored well]

### Priority Improvements

1. [Criterion] — [What is wrong] — [Specific fix]
2. [Criterion] — [What is wrong] — [Specific fix]
3. [Criterion] — [What is wrong] — [Specific fix]
```

---

## Cross-Verification Checklist (for AI auditors)

When auditing as an AI assistant, perform these verification steps:

- [ ] Extract documented symbols from README code blocks first: imports, function/class names, CLI commands/flags, config keys, env vars, and file paths
- [ ] Compare every import/require statement in README examples against the project's actual public exports (manifest exports, `__all__`, index files, type declarations)
- [ ] Compare every CLI command in the README against the project's actual CLI definition (bin field, argument parser)
- [ ] For renamed or missing APIs, name both the stale README symbol and the source-backed current symbol; cite the source file that proves the correction
- [ ] Do not invent aliases, compatibility wrappers, or implementation details for README examples when source does not prove them
- [ ] Check that the package name in install commands matches the actual published package name
- [ ] Verify that configuration keys/options in examples exist in the project's config schema or types
- [ ] Check that file paths referenced in the README (e.g., `src/config.js`) actually exist
- [ ] Confirm the LICENSE file content matches what the README states
- [ ] If a version number appears in the README, check it against the latest release/tag
- [ ] For READMEs with badges, verify badge URLs use the correct repo owner/name
