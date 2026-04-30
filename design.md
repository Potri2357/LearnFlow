# LearnFlow UI/UX Design Specification

Date: 2026-04-23
Product: LearnFlow
Design Direction: Premium light-first
Scope: End-to-end product UI redesign blueprint for Phase 1 implementation

## 1. Goals

### Primary goals

- Deliver a cohesive, premium learning experience across all major flows.
- Improve comprehension and actionability on data-heavy pages.
- Reduce friction from landing -> onboarding -> daily learning loops.
- Standardize components, state handling, and visual hierarchy.

### Non-goals (Phase 1)

- Backend behavior changes or model changes.
- New AI capability expansion outside current API contracts.
- Feature removal or route architecture rewrites.

## 2. Product Experience Principles

- Learning-first clarity: every screen must answer "what should I do next?"
- Guided progression: each flow has clear step order and checkpoints.
- Fast confidence: users should perceive speed through progressive rendering.
- Calm intelligence: visually rich but not noisy; high signal density.
- Consistent trust: same controls, states, and language across modules.

## 3. Information Architecture

### Top-level journey grouping

1. Capture

- Lectures
- Summarize

2. Practice

- Quiz
- Flashcards
- Weak Topics

3. Analyze

- Dashboard
- Quiz Result

4. Plan

- Study Plan

5. Exam

- Exam Preparation

6. Coach

- Concept Coach

7. Account

- Profile
- Notifications

### Route coverage map

- `/` Landing page
- `/login`, `/register`, `/google-login`, `/auth/google/callback`
- `/dashboard`
- `/lectures`
- `/quiz`, `/quiz/:noteId`, `/quiz-mode`
- `/weak-topics`, `/weak-topics/:noteId`
- `/study-plan`, `/analysis`
- `/flashcards`
- `/summarize`
- `/exam-preparation`
- `/concept-coach`
- `/profile`
- `/quiz-result`

## 4. Design System Specification

### 4.1 Color tokens

Use existing brand palette as base, tighten usage rules:

- `brand.primary`: `#2563EB` (CTAs, active nav, primary metrics)
- `brand.secondary`: `#7C3AED` (AI/coach accents)
- `brand.success`: `#10B981` (completed, mastery, positive states)
- `brand.warning`: `#F59E0B` (exam urgency, cautions)
- `brand.error`: `#F43F5E` (destructive, critical errors)
- `brand.info`: `#06B6D4` (supporting insights)

Surface tokens:

- `bg.canvas`: `#EEF4FF`
- `bg.elevated`: `#FFFFFF`
- `bg.subtle`: `#F8FBFF`
- `text.primary`: `#13203A`
- `text.secondary`: `#41516D`
- `border.default`: `#DCE7FF`

### 4.2 Typography

- Family: Lexend (headings), Noto Sans (body)
- Hierarchy:
  - Display: 48/56, 900
  - H1: 40/48, 800
  - H2: 32/40, 800
  - H3: 26/34, 700
  - H4: 22/30, 700
  - Body L: 16/26, 400
  - Body M: 14/22, 400
  - Caption: 12/18, 500

### 4.3 Spacing and layout

- Base spacing unit: 8px
- Content width behavior:
  - Mobile: 100% with 16px gutters
  - Tablet: 100% with 24px gutters
  - Desktop: max-content region 1200-1320px
- Section rhythm:
  - Hero to section: 72px desktop, 48px mobile
  - Card group vertical spacing: 24px

### 4.4 Shape and elevation

- Radius scale: 8, 12, 16, 20
- Core card radius: 20
- Inputs and small controls: 12
- Shadow levels:
  - `shadow.1`: subtle hoverless surfaces
  - `shadow.2`: interactive cards
  - `shadow.3`: overlays and spotlight components

### 4.5 Motion

- Timing:
  - Micro interactions: 120-180ms
  - Layout transitions: 220-300ms
- Easing: cubic-bezier(0.22, 1, 0.36, 1)
- Rules:
  - Motion explains hierarchy, not decoration.
  - Keep one major attention animation per viewport section.

## 5. Global Shell Redesign

## Sidebar

- Keep current fixed desktop + drawer mobile pattern.
- Reorder items by journey grouping:
  - Dashboard
  - Lectures
  - Summarize
  - Quiz
  - Weak Topics
  - Flashcards
  - Study Plan
  - Exam Preparation
  - Concept Coach
  - Profile
- Add missing color mappings for weak-topics and quiz-result states.
- Add compact badges for flagship areas: Concept Coach, Exam Prep.

## Top app bar

- Left: page title + optional breadcrumb subtitle.
- Right: notifications, language switcher, user avatar.
- On mobile, keep sticky app bar with drawer trigger and title.

## Global utilities

- Language switch remains in shell.
- Introduce optional font-size quick control in shell footer.
- Standardize empty/error banners with reusable component.

## 6. Cross-Page UX Standards

Every feature page must implement these states:

- Loading: skeleton first, spinner only for blocking actions.
- Empty: illustration + plain-language prompt + one primary CTA.
- Error: non-technical explanation + retry and fallback action.
- Success feedback: inline confirmation for non-destructive actions.

Form standards:

- Label always visible.
- Helper text before validation errors.
- Validation on blur + on submit.
- Action buttons maintain fixed placement near primary context.

## 7. Page-by-Page Design Specifications

## 7.1 Landing (`/`)

Objectives:

- Clarify value proposition in under 8 seconds.
- Drive users to register/login.

Structure:

1. Hero with headline, subheadline, dual CTAs (Get Started, Login)
2. Core value proof strip (adaptive quiz, exam prep, concept coach)
3. How it works in 3 steps
4. Feature grid with category framing
5. Trust section (privacy, speed, AI guidance quality)
6. Final CTA band

Improvements:

- Reduce visual noise in animated background on small screens.
- Keep one dominant CTA color and one secondary style.
- Add mobile-first spacing rhythm.

## 7.2 Auth (`/login`, `/register`)

Objectives:

- Decrease login confusion and failed attempts.
- Keep social login visually equivalent to email path.

Structure:

- Left panel: context and benefits
- Right panel: auth form card
- Inline password rules and clear error messaging

## 7.3 Dashboard (`/dashboard`)

Objectives:

- Make daily priorities obvious.
- Improve readability of analytics blocks.

Layout:

1. KPI strip (study time, score, streak, mastery)
2. Priority panel: top weak topics + next actions
3. Trend and performance charts
4. Recent activity feed

Improvements:

- Skeletons for KPI + chart placeholders.
- Standardized card header anatomy (title, supporting metric, action).

## 7.4 Lectures (`/lectures`)

Objectives:

- Make upload and note management frictionless.

Layout:

1. Upload area (dropzone + supported formats)
2. Notes list/grid toggle
3. Per-note actions (view, summarize, generate quiz, delete)

Improvements:

- Add first-use onboarding panel for empty library.
- Keep long processing states with progress messaging.

## 7.5 Quiz setup and play (`/quiz`, `/quiz/:noteId`, `/quiz-mode`)

Objectives:

- Speed from setup to answering.
- Make progress and time awareness clear.

Layout:

- Setup page: note selector, difficulty intent, question count.
- Quiz mode: question card, options, confidence actions, progress rail.

Improvements:

- Keyboard-friendly option selection.
- Sticky progress + question index.
- Clear feedback transitions between questions.

## 7.6 Weak Topics (`/weak-topics`, `/weak-topics/:noteId`)

Objectives:

- Convert weakness data into actionable learning steps.

Layout:

- Topic cards sorted by weakness score.
- Each card: weakness trend, recommended action, quick explain CTA.

## 7.7 Study Plan (`/study-plan`, `/analysis`)

Objectives:

- Make generated plans skimmable and executable.

Layout:

- Plan summary bar (duration, hours, focus split)
- Daily/weekly blocks
- Resource and revision blocks

Improvements:

- Export/print-friendly layout.
- "Mark done" and progress snapshots.

## 7.8 Flashcards (`/flashcards`)

Objectives:

- Improve session rhythm and memory reinforcement.

Layout:

- Deck selector
- Card stage
- Response controls (Know, Review, Hard)

Improvements:

- Cleaner flip animation with reduced motion fallback.
- Session stats panel (attempted, confidence distribution).

## 7.9 Summarize (`/summarize`)

Objectives:

- Make long-form summary easy to scan and act on.

Layout:

- Source selector
- Summary sections (key ideas, formulas, action bullets)
- Optional flowchart rendering area

## 7.10 Exam Preparation (`/exam-preparation`)

Objectives:

- Reduce complexity in multi-step exam workflows.

Layout:

1. Stepper: Upload syllabus -> Upload papers -> Configure marks -> Generate
2. Question bank panel with filters
3. Strategy roadmap output panel

Improvements:

- Preserve partial progress locally.
- Surface generation prerequisites clearly.

## 7.11 Concept Coach (`/concept-coach`)

Objectives:

- Elevate flagship tutoring experience.

Layout:

- Chat timeline center
- Left context rail: topic and learning goal
- Right quick-actions rail: hint, formula, explain differently

Improvements:

- Strong distinction between user/assistant cards.
- Listening and thinking states with explicit affordance.
- Quick recap cards after each solved thread.

## 7.12 Profile (`/profile`)

Objectives:

- Make account and progress identity cohesive.

Layout:

- Header profile card
- Learning stats
- Preferences and account actions

## 7.13 Quiz Result (`/quiz-result`)

Objectives:

- Translate outcome into immediate next step.

Layout:

- Score hero
- Topic breakdown and mistakes
- Suggested next actions CTA cluster

## 8. Component Inventory (Build/Refactor)

Create or standardize these shared components:

- `PageHeader`
- `SurfaceCard`
- `MetricCard`
- `EmptyState`
- `ErrorState`
- `LoadingSkeletonPack`
- `SectionTabs`
- `StepProgress`
- `ActionBar`
- `InsightBadge`

## 9. Accessibility Baseline

- Color contrast minimum WCAG AA for text and controls.
- Visible keyboard focus for all interactive elements.
- All icon-only controls require aria-label.
- Reduced-motion mode support for major animations.
- Semantic heading order per page.

## 10. Responsiveness Rules

Breakpoints:

- `xs`: 0-599
- `sm`: 600-899
- `md`: 900-1199
- `lg`: 1200+

Behavior:

- Sidebar becomes drawer under `md`.
- Dense multi-column dashboards collapse progressively.
- Primary CTA remains above fold on mobile critical pages.

## 11. Performance UX Rules

- Use skeletons for any content block loading > 300ms.
- Defer non-critical panels below first viewport.
- Avoid layout shift in chart containers by predefining heights.
- Prefer optimistic UI for fast local feedback where safe.

## 12. API-Aware UI Mapping

- Dashboard uses `/dashboard/stats`, `/profile`, `/recent-weak-topics`.
- Lectures uses `/lectures`, `/upload-pdf`, `/lectures/<id>`.
- Quiz uses `/adaptive/quiz/start`, `/submit-mcq`, `/quiz-completed`.
- Study Plan uses `/study-plan`, `/next-actions`, `/ai-insights/<note_id>`.
- Flashcards uses `/flashcards/generate`.
- Summarize uses `/lectures/<note_id>/summarize`.
- Exam uses exam endpoints under `exam_views.py`.
- Coach uses `/concept-coach` and assignment evaluation endpoint where relevant.

## 13. Implementation Backlog

### Phase A: Foundations

- Finalize token usage in theme and global component overrides.
- Build shared state components and page header system.
- Update shell navigation ordering and mobile drawer behavior.

### Phase B: High-impact surfaces

- Landing + auth redesign
- Dashboard analytics readability pass
- Lectures upload/list flow polish

### Phase C: Learning workflows

- Quiz setup/mode/result consistency
- Weak topics actionability improvements
- Study plan and flashcards interaction upgrades
- Summarize content readability improvements

### Phase D: Advanced modules

- Exam preparation stepper and workflow simplification
- Concept coach flagship conversation UX pass
- Profile and account polish

## 14. Acceptance Criteria

- All routes listed in this document have completed design section coverage.
- Global shell behavior is consistent on mobile, tablet, and desktop.
- Every page has standardized loading/empty/error/success states.
- Accessibility checks pass for keyboard focus and color contrast.
- Visual consistency audit confirms reusable component adoption.

## 15. Risks and Mitigations

- Risk: inconsistent legacy styles across pages.
  - Mitigation: enforce shared primitives before page refactors.

- Risk: heavy animations reduce performance on low-end devices.
  - Mitigation: gate complex motion and provide reduced-motion fallback.

- Risk: endpoint latency impacts perceived quality.
  - Mitigation: skeletons, staged rendering, and explicit async messaging.

## 16. Future-ready Notes (Post Phase 1)

- Add full dark-theme parity with mapped tokens and contrast QA.
- Add personalization presets for learner types.
- Add design QA checklist automation in CI for visual regression snapshots.
