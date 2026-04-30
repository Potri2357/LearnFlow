# LearnFlow — Full Platform Analysis & Improvement Plan

**Created:** 2026-04-30
**Based on:** Complete codebase audit (all 23 pages, backend models, views, URLs)
**Status:** Active Planning

---

## 🔍 Current State Assessment

### ✅ What Is Fully Working

| Feature                                                   | Status                     | Notes                                                                   |
| --------------------------------------------------------- | -------------------------- | ----------------------------------------------------------------------- |
| Dashboard with Recharts (AreaChart, RadarChart, BarChart) | ✅ Done                    | 30s polling, weak topic chips, visibility-change refresh                |
| ConceptCoach (Claude-style interface)                     | ✅ Done                    | Chat history sidebar, voice input, typewriter, i18n                     |
| Flashcards (SM-2 SRS)                                     | ✅ Done                    | Anki-style, 4-button rating, session complete screen, table browser tab |
| ExamPreparation — Question Bank Generator                 | ✅ Done                    | Syllabus upload, previous papers, mark distribution, HY badge           |
| ExamPreparation — AI Strategy Roadmap                     | ✅ Done                    | Day-by-day table, hours/day, exam date config                           |
| QuestionBank page                                         | ✅ Done                    | Subject accordion, Bloom's filter, HY filter, quick-attempt inline      |
| StudyPlan                                                 | ✅ Done                    | Generator + Dashboard, timeline, checkbox items, weak topic bars        |
| Lectures page                                             | ✅ Done                    | PDF viewer, library sidebar, AI insights modal, notes tab               |
| SummarizeLectures                                         | ✅ Done                    | Flowchart, collapsible sections                                         |
| NotesSidebar component                                    | ✅ Done (component exists) | 12,715 bytes — needs integration verification                           |
| StickyNote model                                          | ✅ Done                    | note_type, is_pinned, page_number, source_text                          |
| Question model                                            | ✅ Done                    | blooms_level, is_high_yield, relevance_score, question_type             |
| Flashcard model                                           | ✅ Done                    | SM-2 fields: ease_factor, interval, repetitions, next_review_date       |
| Subject system                                            | ✅ Done                    | SUBJECT_COLORS in theme.js, subject field on LectureNote                |
| AI response cache                                         | ✅ Done                    | AIResponseCache model exists                                            |
| Auth (JWT + Google OAuth)                                 | ✅ Done                    | CustomTokenObtainPairView, GoogleCallback, refresh                      |
| RubricEvaluator page                                      | ✅ Done                    | ~19KB, exists as a page                                                 |

### ⚠️ Gaps & Issues Found

| Issue                                                                                        | Severity | Location                        |
| -------------------------------------------------------------------------------------------- | -------- | ------------------------------- |
| `PDFTextSelector.js` is only 305 bytes — effectively a stub                                  | High     | `components/PDFTextSelector.js` |
| `NotesSidebar.js` exists but integration into `Lectures.js` is unverified                    | High     | `Lectures.js`                   |
| `QuestionBank.js` export is JSON-only (no PDF export)                                        | Medium   | `QuestionBank.js:78-84`         |
| `StudyPlan.js` checkboxes are visual-only — state is not persisted                           | Medium   | `StudyPlan.js:443,467,486`      |
| `Flashcards.js` loads cards per-lecture only — no cross-lecture "due today" view             | Medium   | `Flashcards.js:57-73`           |
| `Dashboard.js` uses blue-dominant color palette (VCOLORS) inconsistent with earth-tone theme | Low      | `Dashboard.js:73-94`            |
| `theme.js` has leftover blue-tinted `MuiOutlinedInput`, `MuiFormLabel`, `MuiTab` overrides   | Low      | `theme.js:446-478`              |
| No dark mode toggle exposed to user (ThemeContext exists but no UI control)                  | Medium   | `context/ThemeContext.js`       |
| `RubricEvaluator.js` is not linked from sidebar navigation                                   | Low      | `layout/SidebarLayout.js`       |
| No `LandingPage` → feature page CTAs for non-logged-in users                                 | Low      | `LandingPage.js`                |
| No loading skeletons on `QuestionBank.js`                                                    | Low      | `QuestionBank.js:86-88`         |
| `VideoGenerator.js` component exists (12,666 bytes) but unclear if linked anywhere           | Medium   | `components/VideoGenerator.js`  |

---

## 🧠 Brainstorm: New Features & Improvements

### A. User Experience Wins (High Impact, Low Effort)

1. **Global Dark Mode Toggle** — `ThemeContext` already supports it; just needs a UI toggle in the top bar/sidebar. Critical for study sessions.

2. **Persistent Checkbox State in StudyPlan Timeline** — Save checkbox state to `localStorage` keyed by `(user_id, plan_timestamp)`. Add a "Clear completed" button. Show a % completion bar.

3. **"Due Today" Flashcard Dashboard Widget** — Query `/flashcards/?due_today=true` and show a "N cards due today" badge on the dashboard and a shortcut card. This makes the SRS feel alive.

4. **Sidebar Badge Counts** — Show unread notification count badge, pending flashcard due count, and weak topic count next to respective nav items.

5. **Keyboard Shortcut Overlay** — Press `?` anywhere to open a modal listing keyboard shortcuts for the current page (Quiz: `A/B/C/D`, Flashcards: `space/←/→/1/2/3/4`, Lectures: `+/-/F`).

6. **QuestionBank → PDF Export** — Replace JSON export with proper PDF export using `jsPDF`. Each question as a formatted card with question, options (masked/revealed), Bloom's tag, and explanation.

7. **Profile Page — Learning Statistics** — Add a "My Stats" section showing: total flashcards reviewed, total quiz attempts, avg score trend (sparkline), best subject, study streak calendar heatmap.

8. **Onboarding Tour** — First-time users see a guided tour (using `react-joyride` or custom) across Dashboard → Lectures → Quiz → Flashcards. Stored in `localStorage`.

---

### B. AI-Powered Enhancements (High Value)

9. **Smart Quiz Recommendations** — After quiz completion, the result screen shows: "Based on your performance, try these topics next" (from weak areas) + a "Study These" button that pre-fills ConceptCoach.

10. **Lecture Summary → Flashcard Pipeline** — In `SummarizeLectures`, add a "Convert to Flashcards" button next to each TL;DR bullet / Key Point. One click creates a flashcard from the content.

11. **ConceptCoach → Add to Flashcards** — In each ConceptCoach response, add an "Add to Flashcards" icon button. Clicking it sends a `POST /flashcards/generate/` with the question + AI response as front/back.

12. **ConceptCoach Context Awareness** — Pass the currently active lecture's title/subject when navigating to ConceptCoach from Lectures page (`/concept-coach?context=Lecture+Title`). Coach auto-loads context.

13. **AI-Powered Note Suggestions** — In the Notes sidebar of Lectures, after saving a note, call the AI: "Given this note excerpt, suggest 2 related exam questions." Display inline below the note.

14. **Adaptive Quiz Difficulty** — Track per-question accuracy from `UserAnswer`. Questions with >80% accuracy are marked "mastered" (green dot). Quiz generator skips mastered questions in adaptive mode.

15. **ExamPrep — Answer Quality Evaluator** — In ExamPreparation, after the user types a free-text answer to an exam question, send it to the AI for scoring vs the model answer. Show score breakdown.

---

### C. UI/UX Polish (Design Quality)

16. **Earth-Tone Consistency Audit** — `Dashboard.js` uses an independent blue/purple/green/pink palette (`VCOLORS`) that conflicts with the established Ink/Teal/Sand/Clay/Coral palette. Remap `VCOLORS` to match theme tokens.

17. **Unified Empty States** — Every page (QuestionBank, Flashcards, WeakTopics) needs a consistent, beautiful empty-state illustration/icon + helpful CTA button. Currently inconsistent.

18. **Micro-Animation Additions:**
    - Lecture library: new upload → slide-in animation
    - Quiz answer selection: ripple + color flash
    - Flashcard flip: already has 3D flip — add subtle shadow depth change
    - Dashboard stat cards: number count-up animation on load

19. **Responsive Mobile Layout** — The Lectures 25/75 split and some dashboard grid layouts break on mobile. Add `xs={12}` fallbacks and a collapsible sidebar drawer for mobile.

20. **Theme Token Cleanup in theme.js** — Remove leftover `MuiOutlinedInput` blue color overrides (lines 439-457) that conflict with earth-tone palette. Standardize all focus states to `#2A9D8F`.

21. **Page Transition Animations** — Add `framer-motion` `AnimatePresence` page transitions (already installed — used in Flashcards). Fade+slide on route change using a wrapper in `SidebarLayout`.

22. **Toast Notification System** — Replace scattered `Snackbar` state (found in ExamPreparation, Dashboard, multiple pages) with a global toast provider via React Context.

---

### D. Feature Completions (Planned but Incomplete)

23. **PDF Text Drag-to-Notes** — `PDFTextSelector.js` is a stub (305 bytes). Implement:
    - `window.getSelection()` on mouse-up in PDF container
    - Floating "Add to Notes" button near selection
    - Click → appends selected text to current note with page reference

24. **Notes Sidebar Integration** — Verify `NotesSidebar.js` is correctly imported and rendered in `Lectures.js`. Ensure note_type, is_pinned, and filter by exam mode all work.

25. **Video Generator** — `VideoGenerator.js` component exists but the `/video/generate/` endpoint is present in urls.py. Wire this into the Lectures AI Insights modal as a "Generate Explainer Video" tab.

26. **RubricEvaluator in Navigation** — `RubricEvaluator.js` (19KB) is a complete page but not linked from the sidebar. Add it to `SidebarLayout.js` nav items.

27. **Flashcard Due-Today Filter** — Backend: add `?due_today=true` query param filter to `get_flashcards` view (filter by `next_review_date <= now()`). Frontend: show "Due Today" section at top of Flashcards page.

28. **Study Plan Persistence** — Extend `StudyPlan` model to store the full plan JSON. Add API endpoint `GET /study-plan/latest/` to restore the last generated plan on page load without re-generating.

---

### E. New Pages / Major Features

29. **📊 Analytics Page** — A dedicated `/analytics` route with:
    - GitHub-style activity heatmap (sessions per day, last 90 days)
    - Accuracy trend line (last 20 quizzes)
    - Subject mastery radar chart (already on Dashboard — move here as full-page)
    - Time-per-subject pie chart
    - Badge showcase (existing `Badge` model)
    - Export report as PDF

30. **🎯 Daily Goal System** — Users set a daily study goal (e.g., 20 flashcards, 1 quiz, 30 min reading). Dashboard shows progress ring. Completed goals trigger a celebration animation + badge.

31. **🤝 Collaborative Notes** — Notes sidebar: "Share Note" button generates a shareable link (readonly) for a note. Backend: add `is_public` and `share_token` to `StickyNote`.

32. **📱 PWA Support** — Add `manifest.json` and service worker for offline caching of the app shell. Allow "Add to Home Screen" on mobile.

33. **🔔 Smart Notification System** — Backend signals already create `Notification` objects. Expand:
    - "You have 5 flashcards due today" — daily at 8am (cron/celery)
    - "You haven't studied [subject] in 7 days"
    - "Exam in 3 days!" (from ExamSyllabus data)

34. **🌍 Language Support Expansion** — i18n infrastructure exists (`i18n/` directory, `useTranslation` in ConceptCoach). Extend translation keys to all pages (currently only ConceptCoach uses it).

---

## 📚 Lecture Workspace 2.0 — Brainstormed Product Direction

This is the strongest opportunity to turn Lectures from a file viewer into a real study workspace. The goal is to make one lecture PDF feel like a structured learning environment where sections, notes, and related PDFs are all connected.

### Core Idea

The lecture area should behave like a **grouped study graph**:

- A PDF is uploaded once, then broken into **sections / chapters / topics**.
- Each section becomes a node in a graph, not just a flat entry in a sidebar.
- Related PDFs, notes, flashcards, and summaries can connect to one or more sections.
- Users can jump from a section to the exact PDF page range, then store notes against that section.

### Section / Group Model

The current lecture sidebar should evolve into a **section organizer**:

- Each PDF can have multiple named sections such as Introduction, Definitions, Examples, Formula Set, Exam Focus, and Revision Notes.
- Sections should support nesting, so a topic can contain subtopics.
- A section can be tagged as **high priority**, **weak area**, **exam-relevant**, or **summary only**.
- PDFs should be selectable from within sections, so a section can point to one or more source files rather than being tied to a single document.

### Graph-Based Categorization

The graph should make lecture relationships visible instead of hidden in plain lists:

- Nodes: PDFs, sections, notes, flashcards, extracted key points.
- Edges: “belongs to”, “references”, “derived from”, “related to”, “revision follow-up”.
- Users can see which PDFs contribute to the same concept and which sections are most connected.
- This can later power recommendations such as “show me all notes connected to this formula section” or “open other PDFs that mention this topic”.

### PDF Viewer Improvements

The PDF viewer should feel like a study tool, not just a preview panel:

- Add a left/right split that keeps the PDF visible while the user writes notes.
- Support page anchors, section markers, and quick jump chips for current chapter, next section, and saved highlights.
- Add a mini outline panel for section navigation.
- Highlight selected text, then let the user attach it to a section or note instantly.
- Support a focused reading mode and a dense review mode.

### Notes Option in Lectures

Notes should become a first-class lecture feature:

- Every section should have an attached notes area.
- Notes can be created as quick thoughts, structured study notes, formula notes, or exam reminders.
- Notes should support pinning, tagging, and linking back to the exact PDF page or selection.
- A note can optionally be converted into a flashcard or a Concept Coach prompt.
- The notes panel should feel lightweight enough for fast capture but powerful enough for revision.

### Better Study Flow

The intended workflow becomes:

1. Upload or open a lecture PDF.
2. Split it into sections and optionally connect related PDFs.
3. Read inside the viewer, select important content, and save it into section notes.
4. Use the graph to revisit related concepts across all lecture files.
5. Promote the best notes into flashcards, summaries, or revision lists.

### What This Unlocks

- A richer lecture organization layer across all PDFs.
- Better revision because topics are grouped by concept rather than by upload order.
- Faster note capture during reading.
- A path toward semantic search later, where users can ask for “all PDFs and notes related to this topic”.
- A more polished study experience that ties Lectures, Notes, Flashcards, and AI summaries together.

### Suggested Acceptance Criteria

- A lecture can be broken into named sections.
- Each section can contain notes and related PDFs.
- Users can select PDF text and attach it to a section note.
- The viewer shows section navigation and a clearer study layout.
- The graph view makes related lecture content discoverable across uploads.

### Likely File Impact

- `frontend/src/pages/Lectures.js` — section UI, notes entry points, viewer layout
- `frontend/src/components/NotesSidebar.js` — richer note capture and section linking
- `frontend/src/components/PDFTextSelector.js` — selection-to-note workflow
- `frontend/src/components/VideoGenerator.js` — optional future tie-in for section explainers
- `core/models.py` — section and relationship storage if the graph becomes persistent
- `core/views.py` — section, note, and related-PDF endpoints

---

## 🤖 Concept Coach 2.0 — Brainstormed Improvements

Concept Coach is already a flagship feature, so the next step is to make it feel like a real tutor instead of a generic chat interface. The best improvements are the ones that improve context, teaching style, follow-up actions, and the ability to return to a session later without losing the learning thread.

### Core Experience Direction

The coach should behave like an adaptive teacher that can switch between explanation, questioning, revision, and exam practice:

- Start with a clear teaching goal, not just a free-form chat prompt.
- Keep the current subject, lecture, or exam topic visible as conversation context.
- Make the response style adjustable: concise, detailed, Socratic, exam-focused, or quick revision.
- Preserve the thread so a learner can return later without losing the learning path.

### Smarter Context Awareness

The coach should always know what the learner is working on:

- Pull in active lecture, weak topic, flashcard deck, or exam prep context automatically.
- Allow the user to pin a session context like a subject, chapter, or question set.
- Show which source triggered the current answer so the learner trusts the context.
- Support a visible breadcrumb such as `Physics > Thermodynamics > First Law` or `Lecture 4 > Page 12`.

### Better Teaching Modes

Different study moments need different coaching styles:

- **Explain mode** for full conceptual breakdowns.
- **Hint mode** for nudging without revealing the answer.
- **Quiz me mode** for active recall and mini-assessments.
- **Exam mode** for concise, high-yield, point-wise responses.
- **Review mode** for short summaries and rapid refresh before tests.

### Response Quality Improvements

The output should feel more structured and easier to scan:

- Break long responses into definition, intuition, example, and recap.
- Add a “key takeaway” line at the end of each answer.
- Highlight formulas, mistakes, and exam cues consistently.
- Keep lists short and actionable instead of producing dense walls of text.
- Make the assistant explicitly say when it is uncertain or needs more context.

### Learning Actions After Every Answer

Each response should lead to a next step:

- Add buttons for “Turn into flashcard”, “Save as note”, “Ask a follow-up”, and “Quiz me on this”.
- Let the user promote a good explanation into Study Plan or Exam Prep.
- Offer one-click actions to jump back to the originating lecture or weak topic.
- Generate a lightweight recap card after long answers so the learner can revisit the core idea.

### Conversation Memory and Session Flow

The coach should feel persistent and recoverable:

- Keep a short-term session summary so long conversations do not drift.
- Store pinned questions or important explanations inside the thread.
- Support named sessions such as “Chapter 3 revision” or “Exam prep for control systems”.
- Add a conversation timeline that makes it easy to resume the last meaningful point.

### Voice and Multimodal Study Support

Concept Coach can become more natural with richer input modes:

- Improve voice input with clear recording state and transcript review.
- Support image or screenshot-based questions in the future for diagrams and handwritten notes.
- Let users paste a chunk of lecture text and ask for simplification or exam framing.
- Add a read-aloud mode for responses when the user wants passive review.

### Trust and Feedback Features

Students need to know when the answer is helpful and when it may need verification:

- Show source-linked context for answers generated from lecture material.
- Surface confidence or “best effort” indicators where the response is synthesized.
- Add feedback buttons beyond thumbs up/down, such as “too long”, “too short”, “still unclear”, or “wrong context”.
- Let the assistant explain why it chose a certain hint or answer path.

### Study Flow Integration Ideas

Concept Coach should connect cleanly to the rest of the platform:

- Start a conversation from a lecture highlight, weak topic chip, or quiz mistake.
- Use quiz errors to auto-create a focused coaching thread.
- Let Study Plan suggestions open directly inside a coaching session.
- Allow saved coach responses to become flashcards, notes, or revision prompts.

### Suggested Acceptance Criteria

- The coach can run in multiple modes such as explain, hint, quiz, and exam.
- Current lecture or topic context is visible and can be changed quickly.
- Responses include structured sections and a short summary.
- The user can promote an answer into a flashcard, note, or follow-up quiz.
- Conversation state can be resumed later with preserved context.

### Likely File Impact

- `frontend/src/pages/ConceptCoach.js` — mode switching, context header, response actions
- `frontend/src/components/ChatMessage.js` or equivalent — richer response layout and feedback controls
- `frontend/src/context/ChatContext.js` or similar — session memory and thread summary support
- `frontend/src/pages/Lectures.js` — send active lecture context into the coach
- `frontend/src/pages/QuizResult.js` — jump into coach from mistakes and weak areas
- `frontend/src/pages/WeakTopics.js` — open coach with a selected weak topic
- `core/views.py` — context-aware prompt assembly and saved session support if needed

---

## 📋 Implementation Priority

### 🟢 Phase 1 — Quick Wins (1–2 days each, high impact)

| Task                                                   | Files                                    | Effort |
| ------------------------------------------------------ | ---------------------------------------- | ------ |
| **1. Dark Mode Toggle**                                | `SidebarLayout.js`, `ThemeContext.js`    | 2h     |
| **2. StudyPlan checkbox persistence**                  | `StudyPlan.js`                           | 2h     |
| **3. Sidebar badge counts** (notifications, due cards) | `SidebarLayout.js`, new API call         | 3h     |
| **4. QuestionBank PDF export** (jsPDF)                 | `QuestionBank.js`                        | 4h     |
| **5. Earth-tone color consistency audit**              | `Dashboard.js`, `theme.js`               | 4h     |
| **6. Global toast provider**                           | New `ToastContext.js`, all pages         | 4h     |
| **7. RubricEvaluator in sidebar nav**                  | `SidebarLayout.js`, `App.js`             | 1h     |
| **8. Keyboard shortcut overlay** (`?` key)             | New `ShortcutsModal.js`, Quiz/Flashcards | 3h     |

### 🔵 Phase 2 — Feature Completions (2–4 days each)

| Task                                                 | Files                                                  | Effort |
| ---------------------------------------------------- | ------------------------------------------------------ | ------ |
| **9. Flashcard Due-Today filter + dashboard widget** | `Flashcards.js`, `views.py`, `Dashboard.js`            | 1 day  |
| **10. PDF Text Drag-to-Notes** (finish stub)         | `PDFTextSelector.js`, `Lectures.js`, `NotesSidebar.js` | 2 days |
| **11. ConceptCoach → Add to Flashcards**             | `ConceptCoach.js`, API call                            | 4h     |
| **12. Summary → Flashcard pipeline**                 | `SummarizeLectures.js`                                 | 4h     |
| **13. Study Plan persistence**                       | `StudyPlan.js`, `views.py`, `models.py`                | 1 day  |
| **14. Smart quiz recommendations on result**         | `QuizResult.js`                                        | 4h     |
| **15. Onboarding tour**                              | New `OnboardingTour.js`                                | 1 day  |
| **16. VideoGenerator wired into Lectures modal**     | `Lectures.js`, `VideoGenerator.js`                     | 4h     |

### 🟣 Phase 3 — New Features (3–5 days each)

| Task                                | Files                                            | Effort |
| ----------------------------------- | ------------------------------------------------ | ------ |
| **17. Analytics page** `/analytics` | New `Analytics.js`, `views.py` endpoint          | 3 days |
| **18. Daily Goal System**           | `Dashboard.js`, new `UserGoal` model, `views.py` | 2 days |
| **19. Profile learning statistics** | `Profile.js`, `views.py`                         | 2 days |
| **20. Page transition animations**  | `SidebarLayout.js`, framer-motion                | 1 day  |
| **21. Adaptive quiz difficulty**    | `Quiz.js`, `views.py`                            | 2 days |
| **22. ExamPrep answer evaluator**   | `ExamPreparation.js`, `ai_utils.py`              | 2 days |
| **23. Mobile responsive audit**     | All pages                                        | 2 days |

### ⚫ Phase 4 — Advanced / Infrastructure

| Task                                    | Effort |
| --------------------------------------- | ------ |
| Smart notification system (Celery)      | 3 days |
| PWA support (manifest + service worker) | 2 days |
| i18n expansion to all pages             | 3 days |
| Collaborative notes (share_token)       | 2 days |
| Language support expansion              | 3 days |

---

## 🗃️ Backend Changes Required

| Change                                         | Priority | Status                                       |
| ---------------------------------------------- | -------- | -------------------------------------------- |
| `GET /flashcards/?due_today=true` filter       | High     | ❌ Not implemented                           |
| `GET /study-plan/latest/` — restore last plan  | Medium   | ❌ Not implemented                           |
| `POST /study-plan/` — persist full plan JSON   | Medium   | Partially done (model exists, no save logic) |
| `UserGoal` model + CRUD endpoints              | Medium   | ❌ Not implemented                           |
| `StickyNote` — add `is_public`, `share_token`  | Low      | ❌ Not implemented                           |
| Extend `UserAnswer` tracking for adaptive quiz | Medium   | Model exists, needs query logic              |
| `GET /analytics/summary/` — aggregated stats   | Medium   | ❌ Not implemented                           |
| Notification auto-generation (cron/signals)    | Low      | Signals file exists, needs expansion         |
| `Badge` auto-award on milestone events         | Low      | Model exists, no award logic                 |

---

## 🎨 Design System Fixes

### theme.js Inconsistencies to Fix

```
Lines 270–272 (MuiPaper elevation shadows): uses blue rgba(37,99,235,...) — change to rgba(38,70,83,...)
Lines 439–456 (MuiOutlinedInput): focused border uses #2563EB (blue) — change to #2A9D8F (teal)
Lines 453–457 (MuiFormLabel focused): color #2563EB — change to #2A9D8F
Lines 467–478 (MuiTab): .Mui-selected color #2563EB — change to #2A9D8F
Line 488 (MuiTableCell root): borderColor #DCE7FF — change to #E8DCC9
Line 392 (MuiDivider): borderColor #DCE7FF — change to #E8DCC9
```

### Dashboard.js VCOLORS Palette Remap

```js
// Current (blue-dominant, inconsistent):
streak:    linear-gradient(135deg,#F97316,#EF4444)   ← OK (coral/flame)
questions: linear-gradient(135deg,#6366F1,#8B5CF6)   ← CHANGE to teal/ink
mastered:  linear-gradient(135deg,#10B981,#06B6D4)   ← CHANGE to teal gradient
score:     linear-gradient(135deg,#F59E0B,#EC4899)   ← CHANGE to sand/clay

// Proposed (earth-tone aligned):
streak:    linear-gradient(135deg,#E76F51,#F4A261)   // coral → clay
questions: linear-gradient(135deg,#264653,#2A9D8F)   // ink → teal (TIDE)
mastered:  linear-gradient(135deg,#2A9D8F,#E9C46A)   // teal → sand (DUNE)
score:     linear-gradient(135deg,#F4A261,#E76F51)   // clay → coral (SUNSET)
```

---

## ✅ Acceptance Criteria

| Feature                   | Done When                                                                       |
| ------------------------- | ------------------------------------------------------------------------------- |
| Dark mode toggle          | Accessible from sidebar/topbar; persists across sessions                        |
| StudyPlan checkboxes      | Checked state persists in localStorage; reset button clears it                  |
| Sidebar badge counts      | Notification dot and "N due" flashcard count visible in nav                     |
| QuestionBank PDF export   | Downloads formatted PDF with questions + answers toggle                         |
| Earth-tone consistency    | No blue (#2563EB, #6366F1) left in theme or Dashboard VCOLORS                   |
| Flashcard due-today       | `/flashcards/?due_today=true` works; Dashboard shows "N due today" widget       |
| PDF Text Drag-to-Notes    | Select text in PDF → "Add to Notes" button appears → creates note with page ref |
| ConceptCoach → Flashcard  | "Add to Flashcard" button in each coach response creates a card                 |
| Analytics page            | Activity heatmap, accuracy trend, radar chart, badge showcase all render        |
| Daily Goal System         | User sets goal; Dashboard shows progress ring; animation on completion          |
| Toast notification system | All Snackbar calls replaced by global `toast.success()` / `toast.error()`       |
| RubricEvaluator in nav    | Accessible from sidebar, route works                                            |
| VideoGenerator wired      | "Generate Explainer Video" tab in Lectures AI Insights modal                    |
| Study Plan persistence    | Last plan restored on page load; no need to regenerate                          |

---

## 📁 File Impact Summary

| File                                             | Changes Needed                                             |
| ------------------------------------------------ | ---------------------------------------------------------- |
| `frontend/src/theme.js`                          | Fix blue-tinted overrides → earth-tone                     |
| `frontend/src/pages/Dashboard.js`                | Remap VCOLORS, add Due-Today widget                        |
| `frontend/src/pages/Flashcards.js`               | Add due-today filter + section                             |
| `frontend/src/pages/StudyPlan.js`                | Persist checkbox state + load last plan                    |
| `frontend/src/pages/QuestionBank.js`             | Replace JSON export with jsPDF                             |
| `frontend/src/pages/ConceptCoach.js`             | Add "Add to Flashcard" button per response                 |
| `frontend/src/pages/SummarizeLectures.js`        | Add "Convert to Flashcard" on key points                   |
| `frontend/src/pages/QuizResult.js`               | Add smart recommendations section                          |
| `frontend/src/layout/SidebarLayout.js`           | Dark mode toggle, badge counts, RubricEvaluator link       |
| `frontend/src/components/PDFTextSelector.js`     | Full implementation (currently stub)                       |
| `frontend/src/App.js`                            | Add `/analytics` route, `/rubric-evaluator` route          |
| New: `frontend/src/pages/Analytics.js`           | Full analytics page                                        |
| New: `frontend/src/context/ToastContext.js`      | Global toast system                                        |
| New: `frontend/src/components/ShortcutsModal.js` | Keyboard shortcut overlay                                  |
| New: `frontend/src/components/OnboardingTour.js` | First-time user tour                                       |
| `core/views.py`                                  | Due-today filter, latest-plan endpoint, analytics endpoint |
| `core/models.py`                                 | UserGoal model                                             |
| `core/urls.py`                                   | New endpoint URLs                                          |
