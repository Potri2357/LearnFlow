// frontend: src/pages/StudyPlanInteractive.js
import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import API from "../api/api"; // your axios instance
import {
  Container,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  Box,
  Divider,
  Grid,
  TextField,
  InputAdornment,
  IconButton,
  Tooltip,
  CircularProgress,
} from "@mui/material";

import PsychologyIcon from "@mui/icons-material/Psychology";
import MenuBookIcon from "@mui/icons-material/MenuBook";
import TaskAltIcon from "@mui/icons-material/TaskAlt";
import AutorenewIcon from "@mui/icons-material/Autorenew";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import ClearIcon from "@mui/icons-material/Clear";
import LectureSelect from "../components/LectureSelect";

export default function StudyPlanInteractive() {
  const [noteId, setNoteId] = useState("");
  const [planText, setPlanText] = useState("");
  const [strengths, setStrengths] = useState({});
  const [sections, setSections] = useState({
    strengths: "",
    weak: "",
    resources: "",
    practice: "",
    revision: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeStep, setActiveStep] = useState(0);
  const location = useLocation();

  // Auto-load noteId from URL or state and generate plan
  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const noteIdParam = searchParams.get("noteId") || location.state?.noteId;

    if (noteIdParam) {
      setNoteId(noteIdParam);
      // Auto-generate plan
      generatePlan(noteIdParam);
    }
  }, [location.search, location.state]);

  const steps = [
    "Strength Topics",
    "Weak Topics",
    "Recommended Resources",
    "Practice Plan",
    "Revision Plan",
  ];

  const icons = [
    <PsychologyIcon color="error" />,
    <MenuBookIcon color="primary" />,
    <TaskAltIcon color="success" />,
    <AutorenewIcon color="warning" />,
    <AnalyticsIcon color="secondary" />,
  ];

  const sectionColors = [
    { bg: "#f0f9ff", border: "#38bdf8" }, // strengths
    { bg: "#fff7ed", border: "#f59e0b" }, // weak
    { bg: "#f0fdf4", border: "#34d399" }, // resources
    { bg: "#fff1f2", border: "#fb7185" }, // practice
    { bg: "#f8fafc", border: "#60a5fa" }, // revision
  ];

  const getBullets = (text) => {
    if (!text) return [];
    return text
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0) // include all non-empty lines
      .map((l) => {
        // remove leading bullet markers
        let cleaned = l.replace(/^[-•*+]\s*/, "").trim();
        // filter out subsection headers like 'Articles:', 'Easy:', etc.
        if (
          cleaned
            .toLowerCase()
            .match(/^(articles|videos|explanations|easy|medium|hard):\s*$/i)
        ) {
          return "";
        }
        return cleaned;
      })
      .filter((l) => l.length > 0);
  };

  // Parse resource groups (Articles / Videos / Explanations)
  const parseResources = (text) => {
    const lines = (text || "")
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    const groups = { Articles: [], Videos: [], Explanations: [] };
    let current = null;
    lines.forEach((l) => {
      const low = l.toLowerCase();
      if (low.startsWith("articles")) {
        current = "Articles";
        return;
      }
      if (low.startsWith("videos")) {
        current = "Videos";
        return;
      }
      if (
        low.startsWith("explanations") ||
        low.startsWith("short explanations")
      ) {
        current = "Explanations";
        return;
      }
      // collect any non-empty, non-heading line under current group
      if (current && l.length > 0) {
        // remove bullet markers (-, •, *, +)
        const cleaned = l.replace(/^[-•*+]\s*/, "").trim();
        if (cleaned && !cleaned.match(/^(articles|videos|explanations):/i)) {
          groups[current].push(cleaned);
        }
      }
    });
    return groups;
  };

  const generatePlan = async (id) => {
    const targetId = typeof id === "string" ? id : noteId;
    if (!targetId) {
      alert("Enter a lecture note ID");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await API.post("study-plan/", { note_id: targetId });
      const data = res.data || {};

      // Backwards/forwards compatible mapping: backend may return `plan_sections` or `sections`.
      const sectionsPayload =
        data.plan_sections || data.sections || data.planSections || {};

      setPlanText(data.plan || "");
      setStrengths(data.strengths || {});
      // If the backend did not provide parsed sections, fallback to using weak_topics
      if (Object.keys(sectionsPayload).length) {
        setSections(sectionsPayload);
      } else if (data.weak_topics || data.weakTopics) {
        const weakObj = data.weak_topics || data.weakTopics || {};
        const weakLines = Object.keys(weakObj)
          .map((t) => `- ${t}: ${weakObj[t]}`)
          .join("\n");

        const strengthsObj = data.strengths || {};
        const strengthsLines = Object.keys(strengthsObj)
          .map((t) => `- ${t}: ${strengthsObj[t]}`)
          .join("\n");

        setSections({
          strengths: strengthsLines,
          weak: weakLines,
          resources: "",
          practice: "",
          revision: "",
        });
      } else {
        setSections({
          strengths: "",
          weak: "",
          resources: "",
          practice: "",
          revision: "",
        });
      }
      setActiveStep(0);
      // Debug: log received sections to confirm they're not empty
      // console.log("Received sections:", sectionsPayload);
      // console.log("Resources section:", sectionsPayload.resources);
      // console.log("Practice section:", sectionsPayload.practice);
    } catch (err) {
      console.error("Study plan error:", err);
      console.error("Response data:", err.response?.data);
      setError("Failed to generate study plan. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const strengthsList = getBullets(sections.strengths);
  const weakList = getBullets(sections.weak);
  const practice = sections.practice || "";
  const practiceBlocks = (() => {
    const lines = (practice || "").split("\n");
    const blocks = { Easy: [], Medium: [], Hard: [] };
    let cur = null;
    for (let l of lines) {
      const t = l.trim();
      if (/^easy[:\s]/i.test(t)) {
        cur = "Easy";
        continue;
      }
      if (/^medium[:\s]/i.test(t)) {
        cur = "Medium";
        continue;
      }
      if (/^hard[:\s]/i.test(t)) {
        cur = "Hard";
        continue;
      }
      if (cur && (/^[-•]/.test(t) || t.length > 2))
        blocks[cur].push(t.replace(/^[-•\s]+/, "").trim());
    }
    return blocks;
  })();

  const resourcesGroups = parseResources(sections.resources);
  const revisionList = getBullets(sections.revision);

  return (
    <Container maxWidth="md" sx={{ mt: 5, mb: 5 }}>
      <Typography
        variant="h3"
        gutterBottom
        sx={{
          fontWeight: "bold",
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          backgroundClip: "text",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          mb: 4,
        }}
      >
        🎓 Study Plan
      </Typography>

      {noteId && (
        <Typography
          variant="subtitle1"
          sx={{
            textAlign: "center",
            mt: -2,
            mb: 4,
            color: "text.secondary",
            fontWeight: "bold",
          }}
        >
          Lecture ID: {noteId}
        </Typography>
      )}

      {loading && !planText && (
        <Container
          maxWidth="md"
          style={{ marginTop: 40, textAlign: "center", marginBottom: 40 }}
        >
          <CircularProgress />
          <Typography variant="h6" sx={{ mt: 2 }}>
            Generating Study Plan for Note #{noteId}...
          </Typography>
        </Container>
      )}

      {error && (
        <Typography color="error" sx={{ mb: 2, textAlign: "center" }}>
          {error}
        </Typography>
      )}

      {!loading && !planText && (
        <Card
          sx={{
            p: 3,
            mb: 4,
            background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
            boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
            borderRadius: 3,
          }}
        >
          <CardContent>
            <Typography
              variant="h6"
              sx={{ mb: 2, fontWeight: 700, color: "#333" }}
            >
              📚 Enter Lecture Note ID
            </Typography>

            <Box
              sx={{
                display: "flex",
                gap: 2,
                alignItems: "stretch",
                flexWrap: { xs: "wrap", sm: "nowrap" },
              }}
            >
              <LectureSelect value={noteId} onChange={(v) => setNoteId(v)} />

              <Button
                variant="contained"
                onClick={() => generatePlan(noteId)}
                disabled={loading || !noteId}
                sx={{
                  height: "56px",
                  minWidth: "180px",
                  whiteSpace: "nowrap",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {loading ? (
                  <CircularProgress size={18} sx={{ color: "white" }} />
                ) : (
                  "Generate Study Plan"
                )}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Strengths numeric mapping (bars) */}
      {Object.keys(strengths || {}).length > 0 && (
        <Card
          sx={{
            p: 3,
            mb: 4,
            background: "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
            boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
            borderRadius: 3,
          }}
        >
          <Typography
            variant="h5"
            sx={{ mb: 3, fontWeight: "bold", color: "#333" }}
          >
            ⭐ Detected Strengths
          </Typography>
          <Grid container spacing={2}>
            {Object.entries(strengths)
              .sort((a, b) => b[1] - a[1])
              .map(([topic, score], idx) => {
                const pct = Math.round(Number(score) * 100);
                return (
                  <Grid item xs={12} sm={6} key={idx}>
                    <Box sx={{ mb: 1, fontWeight: "bold", color: "#333" }}>
                      {topic} — {pct}%
                    </Box>
                    <Box
                      sx={{
                        height: 12,
                        background: "rgba(255,255,255,0.5)",
                        borderRadius: 10,
                        overflow: "hidden",
                      }}
                    >
                      <Box
                        sx={{
                          height: "100%",
                          width: `${pct}%`,
                          background:
                            "linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%)",
                          borderRadius: 10,
                          transition: "width 0.5s ease",
                        }}
                      />
                    </Box>
                  </Grid>
                );
              })}
          </Grid>
        </Card>
      )}

      {/* If plan was produced, show Stepper for the remaining sections */}
      {planText && (
        <>
          <Stepper
            activeStep={activeStep}
            alternativeLabel
            sx={{
              mb: 4,
              "& .MuiStepLabel-label": {
                fontWeight: 600,
                fontSize: "0.9rem",
              },
            }}
          >
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>

          <Card
            sx={{
              p: 4,
              mb: 3,
              backgroundColor: sectionColors[activeStep].bg,
              borderLeft: `8px solid ${sectionColors[activeStep].border}`,
              borderRadius: 3,
              boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
              background: `linear-gradient(135deg, ${sectionColors[activeStep].bg} 0%, rgba(200,200,200,0.05) 100%)`,
              transition: "all 0.3s ease",
            }}
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
                <Box
                  sx={{
                    fontSize: "2rem",
                    mr: 2,
                    background: sectionColors[activeStep].border,
                    color: "white",
                    width: 50,
                    height: 50,
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {icons[Math.max(0, Math.min(icons.length - 1, activeStep))]}
                </Box>
                <Typography
                  variant="h5"
                  sx={{
                    fontWeight: "bold",
                    color: "#333",
                  }}
                >
                  {steps[activeStep]}
                </Typography>
              </Box>
              <Divider sx={{ mb: 3 }} />

              <Grid container spacing={2}>
                {/* Strengths (Step 0) */}
                {activeStep === 0 && (
                  <>
                    {strengthsList.length > 0 ? (
                      strengthsList.map((s, i) => (
                        <Grid item xs={12} key={i}>
                          <Card
                            sx={{
                              p: 2,
                              background: "white",
                              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                              borderLeft: `4px solid ${sectionColors[0].border}`,
                              transition: "all 0.3s ease",
                              "&:hover": {
                                boxShadow: "0 6px 16px rgba(0,0,0,0.12)",
                                transform: "translateY(-2px)",
                              },
                            }}
                          >
                            <Typography sx={{ color: "#333", lineHeight: 1.6 }}>
                              ✓ {s}
                            </Typography>
                          </Card>
                        </Grid>
                      ))
                    ) : (
                      <Grid item xs={12}>
                        <Typography sx={{ color: "#999", fontStyle: "italic" }}>
                          No strength details found in plan text.
                        </Typography>
                      </Grid>
                    )}
                  </>
                )}

                {/* Weak Topics (Step 1) */}
                {activeStep === 1 && (
                  <>
                    {weakList.length > 0 ? (
                      weakList.map((s, i) => (
                        <Grid item xs={12} key={i}>
                          <Card
                            sx={{
                              p: 2,
                              background: "white",
                              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                              borderLeft: `4px solid ${sectionColors[1].border}`,
                              transition: "all 0.3s ease",
                              "&:hover": {
                                boxShadow: "0 6px 16px rgba(0,0,0,0.12)",
                                transform: "translateY(-2px)",
                              },
                            }}
                          >
                            <Typography sx={{ color: "#333", lineHeight: 1.6 }}>
                              ⚠ {s}
                            </Typography>
                          </Card>
                        </Grid>
                      ))
                    ) : (
                      <Grid item xs={12}>
                        <Typography sx={{ color: "#999", fontStyle: "italic" }}>
                          No weak topics found.
                        </Typography>
                      </Grid>
                    )}
                  </>
                )}

                {/* Resources (Step 2) */}
                {activeStep === 2 && (
                  <>
                    {["Articles", "Videos", "Explanations"].map((g) => (
                      <Grid item xs={12} key={g}>
                        <Typography
                          variant="h6"
                          sx={{
                            fontWeight: "bold",
                            color: "#333",
                            mb: 1.5,
                          }}
                        >
                          {g === "Articles" && "📄"} {g === "Videos" && "🎥"}{" "}
                          {g === "Explanations" && "💡"} {g}
                        </Typography>
                        {resourcesGroups[g] && resourcesGroups[g].length > 0 ? (
                          resourcesGroups[g].map((r, i) => (
                            <Card
                              key={i}
                              sx={{
                                p: 1.5,
                                my: 1,
                                background: "white",
                                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                                transition: "all 0.3s ease",
                                "&:hover": {
                                  boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
                                  transform: "translateX(4px)",
                                },
                              }}
                            >
                              <Typography
                                sx={{ color: "#555", fontSize: "0.95rem" }}
                              >
                                • {r}
                              </Typography>
                            </Card>
                          ))
                        ) : (
                          <Typography
                            sx={{ color: "#999", fontStyle: "italic" }}
                          >
                            No {g.toLowerCase()} found.
                          </Typography>
                        )}
                      </Grid>
                    ))}
                  </>
                )}

                {/* Practice Plan (Step 3) */}
                {activeStep === 3 && (
                  <>
                    {["Easy", "Medium", "Hard"].map((lvl) => (
                      <Grid item xs={12} sm={4} key={lvl}>
                        <Card
                          sx={{
                            p: 2,
                            background:
                              lvl === "Easy"
                                ? "linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)"
                                : lvl === "Medium"
                                ? "linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)"
                                : "linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)",
                            borderRadius: 2,
                            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                          }}
                        >
                          <Typography
                            variant="h6"
                            sx={{
                              fontWeight: "bold",
                              color: "#333",
                              mb: 1.5,
                            }}
                          >
                            {lvl === "Easy" && "🟢"}
                            {lvl === "Medium" && "🟡"}
                            {lvl === "Hard" && "🔴"} {lvl}
                          </Typography>
                          {practiceBlocks[lvl] &&
                          practiceBlocks[lvl].length > 0 ? (
                            practiceBlocks[lvl].map((t, i) => (
                              <Typography
                                key={i}
                                sx={{
                                  fontSize: "0.9rem",
                                  color: "#333",
                                  mb: 0.8,
                                }}
                              >
                                • {t}
                              </Typography>
                            ))
                          ) : (
                            <Typography
                              sx={{
                                color: "#999",
                                fontStyle: "italic",
                                fontSize: "0.9rem",
                              }}
                            >
                              No {lvl.toLowerCase()} tasks.
                            </Typography>
                          )}
                        </Card>
                      </Grid>
                    ))}
                  </>
                )}

                {/* Revision (Step 4) */}
                {activeStep === 4 && (
                  <>
                    {revisionList.length > 0 ? (
                      revisionList.map((r, i) => (
                        <Grid item xs={12} key={i}>
                          <Card
                            sx={{
                              p: 2,
                              background: "white",
                              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                              borderLeft: `4px solid ${sectionColors[4].border}`,
                              transition: "all 0.3s ease",
                              "&:hover": {
                                boxShadow: "0 6px 16px rgba(0,0,0,0.12)",
                                transform: "translateY(-2px)",
                              },
                            }}
                          >
                            <Typography sx={{ color: "#333", lineHeight: 1.6 }}>
                              🔄 {r}
                            </Typography>
                          </Card>
                        </Grid>
                      ))
                    ) : (
                      <Grid item xs={12}>
                        <Typography sx={{ color: "#999", fontStyle: "italic" }}>
                          No revision items found.
                        </Typography>
                      </Grid>
                    )}
                  </>
                )}


              </Grid>
            </CardContent>
          </Card>

          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              gap: 2,
              mb: 5,
            }}
          >
            <Button
              variant="outlined"
              disabled={activeStep === 0}
              onClick={() => setActiveStep((s) => Math.max(0, s - 1))}
              sx={{
                fontWeight: 600,
                borderRadius: 2,
                height: "56px",
                px: 4,
                minWidth: "160px",
                fontSize: "1rem",
                borderColor: "#667eea",
                color: "#667eea",
                transition: "all 0.3s ease",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                "&:hover": {
                  borderColor: "#764ba2",
                  color: "#764ba2",
                  background: "rgba(102, 126, 234, 0.04)",
                },
              }}
            >
              ← Previous
            </Button>
            <Button
              variant="contained"
              disabled={activeStep === steps.length - 1}
              onClick={() =>
                setActiveStep((s) => Math.min(steps.length - 1, s + 1))
              }
              sx={{
                fontWeight: 600,
                borderRadius: 2,
                height: "56px",
                px: 4,
                minWidth: "160px",
                fontSize: "1rem",
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                boxShadow: "0 4px 15px rgba(102, 126, 234, 0.4)",
                transition: "all 0.3s ease",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                "&:hover": {
                  boxShadow: "0 6px 20px rgba(102, 126, 234, 0.6)",
                  transform: "translateY(-2px)",
                },
              }}
            >
              Next →
            </Button>
          </Box>
        </>
      )}
    </Container>
  );
}
