---
name: code-animation-earning-reference
description: Use the extracted 341-style code-animation library as a reference when planning or producing faceless short videos for the user's content and earning plan. Routes topics to suitable motion languages, deterministic Canvas/HTML production, audio, rendering, and cross-platform reuse. Do not use it to package or resell the source prompt library.
metadata:
  short-description: Faceless code-animation content reference
---

# Code-animation earning reference

Use this skill when the user asks for faceless short-video ideas, animated explainers, code-generated visuals, or a repeatable content pipeline for the earning plan.

## Source and boundary

The extracted source is at:
`<PROJECT>\reference_materials\code-animation-341-styles`

It contains 341 styles, 193 still references, examples, and licensing notes. Read the source README, `说明.md`, and the relevant style file before using a style. Preserve the source attribution when sharing or citing the library. The source license is custom non-commercial: do not package, sell, or rebrand its prompt library, stills, examples, or direct derivatives as a paid course, skill, product, or service. For monetized videos, prefer self-written code and general motion principles; verify any specific source style or third-party component against `LICENSE`, `NOTICE.md`, and `许可与来源.md` before commercial use.

## Content workflow

1. Start with a shareable question or observation, not a product pitch.
2. Choose one motion language that reinforces the idea:
   - `A2 光标打字`: hesitation, inner monologue, a sentence being revised.
   - `A3 大字冲击`: a claim, punchline, or strong hook.
   - `B3 粒子聚形`: abstract concepts becoming a concrete shape; useful for memory, context, identity, or system state.
   - `F3 分屏对比`: before/after, old/new behavior, or two approaches.
   - `G2 信号故障`: forgetting, interruption, corruption, or a broken context.
3. Write a short deterministic timeline: duration, beats, camera movement, text changes, and ending frame.
4. Implement a self-contained HTML/Canvas animation with seeded randomness. The same time value must render the same frame. Do not use `Math.random()` inside `draw`; initialize all random parameters once with a seeded generator.
5. Specify audio explicitly. Default deliverables include a music bed plus light SFX; if the user wants silent or self-voiced output, state that explicitly.
6. Render a review frame or direction board before making a full batch. For a new visual design, show three materially different directions and wait for the user's selection.
7. Deliver a vertical social version plus a reusable horizontal or longer cut when practical. Include cover frame, burned-in subtitles or SRT, platform caption, and a short comment prompt.
8. Track performance by hook retention, completion rate, saves, shares, and follows. Do not buy traffic until one format produces repeatable organic retention.

## Business-plan use

Treat the animation library as a production reference, not the product being sold. The monetization path is:

- publish original, useful faceless explainers;
- reuse one core animation across Douyin, Xiaohongshu, Bilibili, Xigua, and (if available) YouTube;
- build a recognizable series around AI memory, independent development, and practical AI failures;
- monetize only through platform revenue, viewer support, sponsorships, or clearly licensed tool affiliations unless the user later chooses a different model.

Keep the user's current constraint: pure sharing, no face on camera, and low-follower accounts. Prefer formats that can be produced from screen recordings, generated motion, captions, and optional voiceover.

## Important implementation note

Some sample snippets in the source are illustrative rather than production-safe. In particular, if a sample uses `Math.random()` or mutates frame state, replace it with seeded initialization and a pure time-based `draw(t)` before reuse. Keep the content idea and motion language, but write the implementation independently.

For style-specific details, read [style-routing.md](references/style-routing.md).
