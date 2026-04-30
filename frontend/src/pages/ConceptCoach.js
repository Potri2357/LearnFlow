import React, { useState, useRef, useEffect, useCallback } from "react";
import { useLocation } from "react-router-dom";
import {
  Box,
  Typography,
  TextField,
  IconButton,
  Stack,
  Avatar,
  Chip,
  CircularProgress,
  useTheme,
  Paper,
  Tooltip,
  Fade,
  Divider,
  Button,
  Drawer,
  Alert,
} from "@mui/material";
import {
  Send as SendIcon,
  Person as PersonIcon,
  SmartToy as SmartToyIcon,
  Lightbulb as LightbulbIcon,
  Add as AddIcon,
  ContentCopy as CopyIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  Refresh as RefreshIcon,
  AutoAwesome as AutoAwesomeIcon,
  School as SchoolIcon,
  Calculate as CalculateIcon,
  Psychology as PsychologyIcon,
  CheckCircleOutline as CheckCircleOutlineIcon,
  Mic as MicIcon,
  Stop as StopIcon,
  History as HistoryIcon,
  Close as CloseIcon,
  BookmarkBorder as BookmarkIcon,
} from "@mui/icons-material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import API from "../api/api";
import { useTranslation } from "react-i18next";

// ─── Helpers ─────────────────────────────────────────────────────────────────

const formatTime = () =>
  new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

// ─── Markdown Renderer ───────────────────────────────────────────────────────
// Uses react-markdown + remark-gfm with custom MUI-styled components

// Context to track whether current list is ordered
const ListTypeContext = React.createContext(false);

// Proper named component so React Hook rules are satisfied
const MdListItem = ({ children }) => {
  const isOrdered = React.useContext(ListTypeContext);
  return (
    <Box display="flex" gap={1.5} sx={{ mb: 0.8, alignItems: "flex-start" }}>
      {isOrdered ? (
        <Box
          sx={{
            minWidth: 24,
            height: 24,
            borderRadius: "50%",
            bgcolor: "primary.main",
            color: "white",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "0.72rem",
            fontWeight: 800,
            flexShrink: 0,
            mt: 0.15,
            // CSS counter driven by the parent ol
            "&::before": { content: "counter(md-ol)" },
            counterIncrement: "md-ol",
          }}
        />
      ) : (
        <Box
          sx={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            bgcolor: "primary.main",
            flexShrink: 0,
            mt: 0.9,
          }}
        />
      )}
      <Box sx={{ flex: 1, lineHeight: 1.8 }}>{children}</Box>
    </Box>
  );
};

const MarkdownContent = React.memo(({ content }) => {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";

  const components = {
    // Paragraphs
    p: ({ children }) => (
      <Typography
        variant="body1"
        sx={{ mb: 1.2, lineHeight: 1.85, "&:last-child": { mb: 0 } }}
      >
        {children}
      </Typography>
    ),
    // Bold
    strong: ({ children }) => (
      <Box component="span" sx={{ fontWeight: 800, color: "text.primary" }}>
        {children}
      </Box>
    ),
    // Italic
    em: ({ children }) => (
      <Box
        component="span"
        sx={{ fontStyle: "italic", color: "text.secondary" }}
      >
        {children}
      </Box>
    ),
    // Inline code
    code: ({ inline, children }) =>
      inline ? (
        <Box
          component="code"
          sx={{
            px: 0.8,
            py: 0.15,
            borderRadius: "5px",
            bgcolor: isDark ? "rgba(37,99,235,0.15)" : "rgba(37,99,235,0.08)",
            color: "primary.main",
            fontFamily: '"Fira Code", "Consolas", monospace',
            fontSize: "0.87em",
            fontWeight: 700,
            border: "1px solid",
            borderColor: isDark ? "rgba(37,99,235,0.3)" : "rgba(37,99,235,0.2)",
          }}
        >
          {children}
        </Box>
      ) : (
        // Block code
        <Box
          sx={{
            my: 1.5,
            borderRadius: 2,
            overflow: "hidden",
            border: "1px solid",
            borderColor: "divider",
          }}
        >
          <Box
            sx={{
              px: 2,
              py: 1,
              bgcolor: isDark ? "rgba(0,0,0,0.4)" : "rgba(0,0,0,0.06)",
              borderBottom: "1px solid",
              borderColor: "divider",
              display: "flex",
              alignItems: "center",
              gap: 1,
            }}
          >
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                bgcolor: "#ef4444",
              }}
            />
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                bgcolor: "#f59e0b",
              }}
            />
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                bgcolor: "#10b981",
              }}
            />
          </Box>
          <Box
            component="pre"
            sx={{
              m: 0,
              p: 2,
              bgcolor: isDark ? "rgba(0,0,0,0.3)" : "rgba(248,250,252,1)",
              fontFamily: '"Fira Code", "Consolas", monospace',
              fontSize: "0.88rem",
              lineHeight: 1.7,
              overflowX: "auto",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            <code>{children}</code>
          </Box>
        </Box>
      ),
    // Headings
    h1: ({ children }) => (
      <Typography
        variant="h5"
        fontWeight={900}
        sx={{ mt: 2, mb: 1, letterSpacing: "-0.01em" }}
      >
        {children}
      </Typography>
    ),
    h2: ({ children }) => (
      <Typography
        variant="h6"
        fontWeight={800}
        sx={{ mt: 1.5, mb: 0.8, color: "primary.main" }}
      >
        {children}
      </Typography>
    ),
    h3: ({ children }) => (
      <Typography
        variant="subtitle1"
        fontWeight={800}
        sx={{ mt: 1.5, mb: 0.5, color: "primary.main" }}
      >
        {children}
      </Typography>
    ),
    h4: ({ children }) => (
      <Typography variant="subtitle2" fontWeight={800} sx={{ mt: 1, mb: 0.5 }}>
        {children}
      </Typography>
    ),
    // Unordered list
    ul: ({ children }) => (
      <ListTypeContext.Provider value={false}>
        <Box component="ul" sx={{ pl: 0, my: 0.5, listStyle: "none" }}>
          {children}
        </Box>
      </ListTypeContext.Provider>
    ),
    // Ordered list — resets CSS counter; MdListItem increments it
    ol: ({ children }) => (
      <ListTypeContext.Provider value={true}>
        <Box
          component="ol"
          sx={{ pl: 0, my: 0.5, listStyle: "none", counterReset: "md-ol" }}
        >
          {children}
        </Box>
      </ListTypeContext.Provider>
    ),
    // Use the extracted proper component
    li: MdListItem,
    // Blockquote — used for hints/tips
    blockquote: ({ children }) => (
      <Box
        sx={{
          my: 1.5,
          pl: 2,
          py: 1,
          borderLeft: "4px solid",
          borderColor: "primary.main",
          bgcolor: isDark ? "rgba(37,99,235,0.07)" : "rgba(37,99,235,0.04)",
          borderRadius: "0 8px 8px 0",
          fontStyle: "italic",
        }}
      >
        {children}
      </Box>
    ),
    // Horizontal rule
    hr: () => <Divider sx={{ my: 2 }} />,
    // Tables (GFM)
    table: ({ children }) => (
      <Box
        sx={{
          my: 1.5,
          overflowX: "auto",
          borderRadius: 2,
          border: "1px solid",
          borderColor: "divider",
        }}
      >
        <Box
          component="table"
          sx={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "0.88rem",
          }}
        >
          {children}
        </Box>
      </Box>
    ),
    thead: ({ children }) => (
      <Box
        component="thead"
        sx={{ bgcolor: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)" }}
      >
        {children}
      </Box>
    ),
    tbody: ({ children }) => <Box component="tbody">{children}</Box>,
    tr: ({ children }) => (
      <Box
        component="tr"
        sx={{
          "&:not(:last-child)": {
            borderBottom: "1px solid",
            borderColor: "divider",
          },
        }}
      >
        {children}
      </Box>
    ),
    th: ({ children }) => (
      <Box
        component="th"
        sx={{
          px: 2,
          py: 1.2,
          textAlign: "left",
          fontWeight: 800,
          whiteSpace: "nowrap",
        }}
      >
        {children}
      </Box>
    ),
    td: ({ children }) => (
      <Box component="td" sx={{ px: 2, py: 1, verticalAlign: "top" }}>
        {children}
      </Box>
    ),
    // Links
    a: ({ href, children }) => (
      <Box
        component="a"
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        sx={{
          color: "primary.main",
          fontWeight: 600,
          textDecoration: "underline",
          textDecorationStyle: "dotted",
        }}
      >
        {children}
      </Box>
    ),
  };

  return (
    <Box
      sx={{ "& > *:first-of-type": { mt: 0 }, "& > *:last-child": { mb: 0 } }}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </Box>
  );
});

// ─── Voice Input Hook ─────────────────────────────────────────────────────────

const useSpeechRecognition = ({ onResult, onError, lang = "en-US" }) => {
  const recognitionRef = useRef(null);
  const [listening, setListening] = useState(false);
  const [supported, setSupported] = useState(false);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      setSupported(true);
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = lang;

      recognition.onresult = (e) => {
        const transcript = Array.from(e.results)
          .map((r) => r[0].transcript)
          .join("");
        onResult(transcript, e.results[e.results.length - 1].isFinal);
      };

      recognition.onerror = (e) => {
        setListening(false);
        onError?.(e.error);
      };

      recognition.onend = () => setListening(false);
      recognitionRef.current = recognition;
    }
  }, [lang, onResult, onError]);

  const start = useCallback(() => {
    if (!recognitionRef.current || listening) return;
    try {
      recognitionRef.current.start();
      setListening(true);
    } catch (e) {
      /* recognition already started */
    }
  }, [listening]);

  const stop = useCallback(() => {
    if (!recognitionRef.current) return;
    recognitionRef.current.stop();
    setListening(false);
  }, []);

  return { listening, supported, start, stop };
};

// ─── Typing indicator ─────────────────────────────────────────────────────────

const TypingIndicator = () => (
  <Box display="flex" gap={0.5} alignItems="center" sx={{ px: 1, py: 0.5 }}>
    {[0, 1, 2].map((i) => (
      <Box
        key={i}
        sx={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          bgcolor: "primary.main",
          animation: "bounce 1.2s ease-in-out infinite",
          animationDelay: `${i * 0.2}s`,
          "@keyframes bounce": {
            "0%, 80%, 100%": { transform: "scale(0.6)", opacity: 0.4 },
            "40%": { transform: "scale(1)", opacity: 1 },
          },
        }}
      />
    ))}
  </Box>
);

// ─── Assistant message bubble ─────────────────────────────────────────────────

const AssistantBubble = ({ msg }) => {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const [thumbed, setThumbed] = useState(null);

  const isError = msg.is_error === true;

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Fade in timeout={400}>
      <Box>
        <Box display="flex" gap={2} alignItems="flex-start">
          <Avatar
            sx={{
              width: 36,
              height: 36,
              background: isError
                ? "linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)"
                : "linear-gradient(135deg, #2563EB 0%, #7c3aed 100%)",
              flexShrink: 0,
            }}
          >
            <SmartToyIcon sx={{ fontSize: 20 }} />
          </Avatar>

          <Box flex={1} minWidth={0}>
            <Typography
              variant="caption"
              fontWeight={700}
              color="text.secondary"
              display="block"
              mb={0.75}
            >
              {t("coach_label")}
            </Typography>

            {/* Main bubble */}
            <Box
              sx={{
                p: 2.5,
                borderRadius: "0 14px 14px 14px",
                bgcolor: isError
                  ? isDark
                    ? "rgba(245,158,11,0.08)"
                    : "rgba(245,158,11,0.05)"
                  : isDark
                    ? "rgba(28, 37, 46, 0.95)"
                    : "rgba(248, 250, 252, 1)",
                border: "1px solid",
                borderColor: isError
                  ? isDark
                    ? "rgba(245,158,11,0.3)"
                    : "rgba(245,158,11,0.25)"
                  : isDark
                    ? "rgba(255,255,255,0.07)"
                    : "rgba(0,0,0,0.07)",
                wordBreak: "break-word",
              }}
            >
              <MarkdownContent content={msg.content} />
            </Box>

            {/* Hint chips */}
            {msg.hints?.length > 0 && (
              <Box mt={1.5} display="flex" flexWrap="wrap" gap={1}>
                {msg.hints.map((hint, i) => (
                  <Chip
                    key={i}
                    icon={<LightbulbIcon />}
                    label={hint}
                    size="small"
                    sx={{
                      bgcolor: "rgba(245,158,11,0.1)",
                      color: "#f59e0b",
                      border: "1px solid rgba(245,158,11,0.25)",
                      fontWeight: 700,
                      borderRadius: 2,
                      "& .MuiChip-icon": { color: "#f59e0b", fontSize: 16 },
                    }}
                  />
                ))}
              </Box>
            )}

            {/* Action row — hide for error messages */}
            {!isError && (
              <Box display="flex" alignItems="center" gap={0.5} mt={0.75}>
                <Typography
                  variant="caption"
                  color="text.disabled"
                  sx={{ mr: 0.5, fontSize: "0.72rem" }}
                >
                  {msg.time}
                </Typography>
                <Tooltip title={copied ? t("coach_copied") : t("coach_copy")}>
                  <IconButton
                    size="small"
                    onClick={handleCopy}
                    sx={{
                      opacity: 0.45,
                      "&:hover": { opacity: 1 },
                      width: 28,
                      height: 28,
                    }}
                  >
                    <CopyIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title={t("coach_helpful")}>
                  <IconButton
                    size="small"
                    onClick={() => setThumbed("up")}
                    sx={{
                      opacity: thumbed === "up" ? 1 : 0.45,
                      color: thumbed === "up" ? "#10b981" : "inherit",
                      width: 28,
                      height: 28,
                      "&:hover": { opacity: 1 },
                    }}
                  >
                    <ThumbUpIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title={t("coach_not_helpful")}>
                  <IconButton
                    size="small"
                    onClick={() => setThumbed("down")}
                    sx={{
                      opacity: thumbed === "down" ? 1 : 0.45,
                      color: thumbed === "down" ? "#ef4444" : "inherit",
                      width: 28,
                      height: 28,
                      "&:hover": { opacity: 1 },
                    }}
                  >
                    <ThumbDownIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            )}
          </Box>
        </Box>
      </Box>
    </Fade>
  );
};

// ─── User message bubble ──────────────────────────────────────────────────────

const UserBubble = ({ msg }) => {
  const { t } = useTranslation();
  return (
    <Fade in timeout={300}>
      <Box
        display="flex"
        gap={2}
        alignItems="flex-start"
        flexDirection="row-reverse"
      >
        <Avatar
          sx={{ width: 36, height: 36, bgcolor: "primary.main", flexShrink: 0 }}
        >
          <PersonIcon sx={{ fontSize: 20 }} />
        </Avatar>
        <Box>
          <Typography
            variant="caption"
            fontWeight={700}
            color="text.secondary"
            display="block"
            mb={0.75}
            textAlign="right"
          >
            {t("coach_you")}
          </Typography>
          <Box
            sx={{
              p: 2,
              borderRadius: "14px 0 14px 14px",
              bgcolor: "primary.main",
              color: "white",
              maxWidth: 520,
            }}
          >
            <Typography
              variant="body1"
              sx={{
                color: "inherit",
                lineHeight: 1.75,
                whiteSpace: "pre-wrap",
              }}
            >
              {msg.content}
            </Typography>
          </Box>
          <Typography
            variant="caption"
            color="text.disabled"
            display="block"
            textAlign="right"
            mt={0.5}
            sx={{ fontSize: "0.72rem" }}
          >
            {msg.time}
            </Typography>
        </Box>
      </Box>
    </Fade>
  );
};

// ─── Starter prompts ──────────────────────────────────────────────────────────

const StarterGrid = ({ onSelect }) => {
  const { t } = useTranslation();
  const starters = [
    {
      icon: <CalculateIcon />,
      labelKey: "starter_math",
      textKey: "starter_math_text",
    },
    {
      icon: <PsychologyIcon />,
      labelKey: "starter_concept",
      textKey: "starter_concept_text",
    },
    {
      icon: <SchoolIcon />,
      labelKey: "starter_check",
      textKey: "starter_check_text",
    },
    {
      icon: <LightbulbIcon />,
      labelKey: "starter_hint",
      textKey: "starter_hint_text",
    },
  ];

  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";

  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: { xs: "1fr 1fr", md: "1fr 1fr 1fr 1fr" },
        gap: 1.5,
        mb: 3,
      }}
    >
      {starters.map((s, i) => (
        <Paper
          key={i}
          onClick={() => onSelect(t(s.textKey))}
          elevation={0}
          sx={{
            p: 2,
            borderRadius: 3,
            cursor: "pointer",
            border: "1px solid",
            borderColor: "divider",
            bgcolor: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
            transition: "all 0.2s",
            "&:hover": {
              borderColor: "primary.main",
              bgcolor: "rgba(37,99,235,0.04)",
              transform: "translateY(-2px)",
              boxShadow: "0 4px 20px rgba(37,99,235,0.1)",
            },
          }}
        >
          <Box sx={{ color: "primary.main", mb: 1, "& svg": { fontSize: 22 } }}>
            {s.icon}
          </Box>
          <Typography variant="body2" fontWeight={700} sx={{ lineHeight: 1.4 }}>
            {t(s.labelKey)}
          </Typography>
        </Paper>
      ))}
    </Box>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────

export default function ConceptCoach() {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";
  const { t, i18n } = useTranslation();
  const location = useLocation();

  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [chatSessions, setChatSessions] = useState(() => {
    try { return JSON.parse(localStorage.getItem('cc_sessions') || '[]'); } catch { return []; }
  });
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const autoExplainFiredRef = useRef(false);

  // Parse URL params for auto-explain from WeakTopics
  const searchParams = new URLSearchParams(location.search);
  const autoExplainTopic = searchParams.get('topic') || '';
  const autoExplainSubject = searchParams.get('subject') || '';
  const shouldAutoExplain = searchParams.get('autoExplain') === 'true';

  // Map i18n language to BCP-47 for speech recognition
  const langMap = { en: "en-US", hi: "hi-IN", ta: "ta-IN", fr: "fr-FR" };
  const speechLang = langMap[i18n.language] || "en-US";

  const handleVoiceResult = useCallback((transcript, isFinal) => {
    setInputValue(transcript);
    if (isFinal && transcript.trim()) {
      // Auto-send on final result
      // (we send after a short delay so the input shows the final text)
      setTimeout(() => {
        setInputValue((prev) => {
          if (prev.trim()) sendMessage(prev.trim());
          return "";
        });
      }, 300);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleVoiceError = useCallback((err) => {
    if (err !== "no-speech") {
      setInputValue("");
    }
  }, []);

  const {
    listening,
    supported: voiceSupported,
    start: startListening,
    stop: stopListening,
  } = useSpeechRecognition({
    onResult: handleVoiceResult,
    onError: handleVoiceError,
    lang: speechLang,
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Auto-explain on mount when coming from WeakTopics
  useEffect(() => {
    if (shouldAutoExplain && autoExplainTopic && !autoExplainFiredRef.current) {
      autoExplainFiredRef.current = true;
      const msg = `Please explain **${autoExplainTopic}**${autoExplainSubject ? ` from ${autoExplainSubject}` : ''} in a clear, comprehensive way with:
1. Simple explanation
2. A concrete example
3. Why this is important for exams
4. A memory tip or mnemonic`;
      // Small delay to let component mount fully
      setTimeout(() => sendMessage(msg), 400);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Save session to localStorage on messages change
  useEffect(() => {
    if (messages.length === 0) return;
    const session = {
      id: Date.now(),
      preview: messages[0]?.content?.slice(0, 60) || 'Session',
      time: new Date().toLocaleDateString(),
      messageCount: messages.length,
    };
    setChatSessions(prev => {
      const updated = [session, ...prev.slice(0, 19)];
      localStorage.setItem('cc_sessions', JSON.stringify(updated));
      return updated;
    });
  }, [messages.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const sendMessage = useCallback(
    async (text) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;

      const userMsg = { role: "user", content: trimmed, time: formatTime() };
      setMessages((prev) => [...prev, userMsg]);
      setInputValue("");
      setLoading(true);
      inputRef.current?.focus();

      try {
        const history = messages
          .slice(-10)
          .map((m) => ({ role: m.role, content: m.content }));
        const res = await API.post("/ai-tutor/chat/", {
          message: trimmed,
          chat_history: history,
        });

        const data = res.data;
        const responseText =
          data.response ||
          data.message ||
          (typeof data === "string" ? data : JSON.stringify(data));
        const hints = Array.isArray(data.hints) ? data.hints : [];
        const isError = !!data.is_error;

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: responseText,
            hints,
            is_error: isError,
            time: formatTime(),
          },
        ]);
      } catch (err) {
        console.error(
          "ConceptCoach API error:",
          err?.response?.data || err.message,
        );
        const errMsg =
          err?.response?.data?.detail ||
          err?.response?.data?.error ||
          err.message ||
          "Unknown error";
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `**Connection error.**\n\nCould not reach the server: \`${errMsg}\`. Please check the backend is running and try again.`,
            hints: [],
            is_error: true,
            time: formatTime(),
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, messages],
  );

  const handleSend = () => sendMessage(inputValue);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setInputValue("");
    inputRef.current?.focus();
  };

  const toggleVoice = () => {
    if (!voiceSupported) {
      alert(t("coach_voice_unsupported"));
      return;
    }
    if (listening) stopListening();
    else startListening();
  };

  const isEmpty = messages.length === 0;

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "calc(100vh - 72px)",
        maxWidth: 980,
        mx: "auto",
        position: "relative",
      }}
    >
      {/* ── HISTORY DRAWER ── */}
      <Drawer
        anchor="left"
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        PaperProps={{ sx: { width: 280, bgcolor: 'background.paper', borderRight: '1px solid', borderColor: 'divider' } }}
      >
        <Box sx={{ p: 2.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="subtitle1" fontWeight={800}>Chat History</Typography>
            <IconButton size="small" onClick={() => setHistoryOpen(false)}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>
          <Button fullWidth variant="contained" startIcon={<AddIcon />}
            onClick={() => { handleNewChat(); setHistoryOpen(false); }}
            sx={{ mb: 2, fontWeight: 700, borderRadius: 2, background: 'linear-gradient(135deg, #2563EB, #7c3aed)' }}>
            New Chat
          </Button>
          {chatSessions.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4, opacity: 0.5 }}>
              <HistoryIcon sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="body2">No history yet</Typography>
            </Box>
          ) : (
            <Stack spacing={0.75}>
              {chatSessions.map((s, i) => (
                <Paper key={i} elevation={0} sx={{
                  p: 1.5, borderRadius: '10px', cursor: 'pointer',
                  border: '1px solid', borderColor: 'divider',
                  '&:hover': { borderColor: 'primary.main', bgcolor: 'rgba(37,99,235,0.04)' },
                  transition: 'all 0.15s',
                }}>
                  <Typography variant="body2" fontWeight={600} noWrap>{s.preview}</Typography>
                  <Typography variant="caption" color="text.disabled">{s.time} · {s.messageCount} msgs</Typography>
                </Paper>
              ))}
            </Stack>
          )}
          {chatSessions.length > 0 && (
            <Button fullWidth size="small" color="error" sx={{ mt: 2, fontWeight: 600 }}
              onClick={() => { setChatSessions([]); localStorage.removeItem('cc_sessions'); }}>
              Clear History
            </Button>
          )}
        </Box>
      </Drawer>

      {/* ── TOP BAR ── */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 2,
          py: 1.5,
          flexShrink: 0,
        }}
      >
        <Box display="flex" alignItems="center" gap={1.5}>
          <Tooltip title="Chat History">
            <IconButton size="small" onClick={() => setHistoryOpen(true)}
              sx={{ borderRadius: 2, border: '1px solid', borderColor: 'divider', width: 36, height: 36 }}>
              <HistoryIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Box
            sx={{
              width: 38, height: 38, borderRadius: "10px",
              background: "linear-gradient(135deg, #2563EB, #7c3aed)",
              display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
            }}
          >
            <SmartToyIcon sx={{ color: "white", fontSize: 20 }} />
          </Box>
          <Box>
            <Typography variant="subtitle1" fontWeight={800} sx={{ lineHeight: 1.2 }}>
              {t("coach_title")}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {t("coach_subtitle")}
            </Typography>
          </Box>
        </Box>
        <Tooltip title={t("coach_new_chat")}>
          <IconButton
            onClick={handleNewChat}
            sx={{
              borderRadius: 2,
              border: "1px solid",
              borderColor: "divider",
              width: 36,
              height: 36,
            }}
          >
            <AddIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      <Divider />

      {/* ── MESSAGES ── */}
      <Box
        sx={{
          flex: 1,
          overflowY: "auto",
          px: { xs: 2, md: 3 },
          py: 3,
          scrollbarWidth: "thin",
          "&::-webkit-scrollbar": { width: 5 },
          "&::-webkit-scrollbar-thumb": { bgcolor: "divider", borderRadius: 3 },
        }}
      >
        {/* Auto-Explain Context Banner */}
        {shouldAutoExplain && autoExplainTopic && (
          <Fade in timeout={500}>
            <Alert
              icon={<SchoolIcon fontSize="inherit" />}
              severity="info"
              sx={{
                mb: 3,
                borderRadius: '12px',
                border: '1px solid',
                borderColor: 'info.main',
                bgcolor: isDark ? 'rgba(2,136,209,0.1)' : 'rgba(2,136,209,0.05)',
                '& .MuiAlert-message': { width: '100%' },
              }}
              action={
                <IconButton size="small" onClick={() => window.history.replaceState({}, document.title, window.location.pathname)}>
                  <CloseIcon fontSize="small" />
                </IconButton>
              }
            >
              <Typography variant="body2" fontWeight={700}>
                Teaching: {autoExplainTopic} {autoExplainSubject ? `(${autoExplainSubject})` : ''}
              </Typography>
            </Alert>
          </Fade>
        )}

        {/* Welcome / empty state */}
        {isEmpty && !shouldAutoExplain && (
          <Fade in timeout={600}>
            <Box>
              <Box textAlign="center" mb={5} mt={2}>
                <Box
                  sx={{
                    width: 72,
                    height: 72,
                    borderRadius: "18px",
                    background:
                      "linear-gradient(135deg, rgba(37,99,235,0.15), rgba(124,58,237,0.15))",
                    border: "2px solid rgba(37,99,235,0.2)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    mx: "auto",
                    mb: 3,
                  }}
                >
                  <AutoAwesomeIcon
                    sx={{ fontSize: 36, color: "primary.main" }}
                  />
                </Box>
                <Typography
                  variant="h4"
                  fontWeight={900}
                  sx={{ letterSpacing: "-0.02em", mb: 1.5 }}
                >
                  {t("coach_welcome_title")}
                </Typography>
                <Typography
                  variant="body1"
                  color="text.secondary"
                  sx={{ maxWidth: 520, mx: "auto", lineHeight: 1.7 }}
                >
                  {t("coach_welcome_subtitle")}
                </Typography>
              </Box>
              <StarterGrid onSelect={setInputValue} />
              <Typography
                variant="caption"
                color="text.disabled"
                display="block"
                textAlign="center"
              >
                {t("coach_welcome_footer")}
              </Typography>
            </Box>
          </Fade>
        )}

        {/* Messages */}
        <Stack spacing={4}>
          {messages.map((msg, idx) =>
            msg.role === "user" ? (
              <UserBubble key={idx} msg={msg} />
            ) : (
              <AssistantBubble key={idx} msg={msg} />
            ),
          )}

          {/* Typing dots */}
          {loading && (
            <Fade in timeout={300}>
              <Box display="flex" gap={2} alignItems="center">
                <Avatar
                  sx={{
                    width: 36,
                    height: 36,
                    background: "linear-gradient(135deg, #2563EB, #7c3aed)",
                    flexShrink: 0,
                  }}
                >
                  <SmartToyIcon sx={{ fontSize: 20 }} />
                </Avatar>
                <Box
                  sx={{
                    px: 2,
                    py: 1.5,
                    borderRadius: "0 14px 14px 14px",
                    bgcolor: isDark
                      ? "rgba(28,37,46,0.95)"
                      : "rgba(248,250,252,1)",
                    border: "1px solid",
                    borderColor: isDark
                      ? "rgba(255,255,255,0.07)"
                      : "rgba(0,0,0,0.07)",
                  }}
                >
                  <TypingIndicator />
                </Box>
              </Box>
            </Fade>
          )}
          <div ref={messagesEndRef} />
        </Stack>
      </Box>

      {/* ── INPUT AREA ── */}
      <Box
        sx={{
          flexShrink: 0,
          px: { xs: 2, md: 3 },
          pb: 3,
          pt: 1.5,
          bgcolor: "background.default",
          borderTop: "1px solid",
          borderColor: "divider",
        }}
      >
        {/* Quick chips */}
        {!isEmpty && (
          <Box display="flex" gap={1} mb={1.5} flexWrap="wrap">
            {[
              {
                labelKey: "chip_hint",
                icon: <LightbulbIcon sx={{ fontSize: 14 }} />,
              },
              {
                labelKey: "chip_formula",
                icon: <CalculateIcon sx={{ fontSize: 14 }} />,
              },
              {
                labelKey: "chip_explain",
                icon: <RefreshIcon sx={{ fontSize: 14 }} />,
              },
              {
                labelKey: "chip_next",
                icon: <CheckCircleOutlineIcon sx={{ fontSize: 14 }} />,
              },
            ].map((chip, i) => (
              <Chip
                key={i}
                icon={chip.icon}
                label={t(chip.labelKey)}
                size="small"
                clickable
                onClick={() => sendMessage(t(chip.labelKey))}
                disabled={loading}
                sx={{
                  borderRadius: 2,
                  fontWeight: 600,
                  fontSize: "0.76rem",
                  bgcolor: isDark
                    ? "rgba(255,255,255,0.04)"
                    : "rgba(0,0,0,0.04)",
                  border: "1px solid",
                  borderColor: "divider",
                  "&:hover": {
                    bgcolor: "rgba(37,99,235,0.09)",
                    borderColor: "primary.main",
                    color: "primary.main",
                  },
                  "& .MuiChip-icon": { color: "inherit" },
                }}
              />
            ))}
          </Box>
        )}

        {/* Voice listening banner */}
        {listening && (
          <Fade in>
            <Box
              sx={{
                mb: 1.5,
                px: 2,
                py: 1,
                borderRadius: 2,
                bgcolor: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.2)",
                display: "flex",
                alignItems: "center",
                gap: 1,
              }}
            >
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  bgcolor: "#ef4444",
                  animation: "pulse 1s ease-in-out infinite",
                  "@keyframes pulse": {
                    "0%,100%": { opacity: 1 },
                    "50%": { opacity: 0.3 },
                  },
                }}
              />
              <Typography variant="caption" fontWeight={700} color="error.main">
                {t("coach_listening")}
              </Typography>
            </Box>
          </Fade>
        )}

        {/* Input box */}
        <Paper
          elevation={0}
          sx={{
            display: "flex",
            alignItems: "flex-end",
            gap: 1,
            p: "10px 14px",
            borderRadius: 4,
            border: "1px solid",
            borderColor: "divider",
            bgcolor: isDark ? "rgba(28,37,46,0.9)" : "white",
            boxShadow: isDark ? "none" : "0 2px 12px rgba(0,0,0,0.07)",
            transition: "border-color 0.2s, box-shadow 0.2s",
            "&:focus-within": {
              borderColor: "primary.main",
              boxShadow: "0 0 0 3px rgba(37,99,235,0.12)",
            },
          }}
        >
          <TextField
            inputRef={inputRef}
            fullWidth
            multiline
            maxRows={6}
            placeholder={
              isEmpty
                ? t("coach_placeholder_empty")
                : t("coach_placeholder_followup")
            }
            variant="standard"
            InputProps={{
              disableUnderline: true,
              sx: { fontSize: "0.96rem", lineHeight: 1.6, pt: 0.5, pb: 0.5 },
            }}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />

          {/* Voice button */}
          <Tooltip title={listening ? t("coach_mic_stop") : t("coach_mic_tip")}>
            <IconButton
              onClick={toggleVoice}
              disabled={loading}
              sx={{
                width: 38,
                height: 38,
                borderRadius: 2,
                flexShrink: 0,
                color: listening ? "#ef4444" : "text.secondary",
                bgcolor: listening ? "rgba(239,68,68,0.1)" : "transparent",
                border: listening
                  ? "1px solid rgba(239,68,68,0.3)"
                  : "1px solid transparent",
                "&:hover": {
                  bgcolor: listening ? "rgba(239,68,68,0.2)" : "action.hover",
                },
                transition: "all 0.2s",
              }}
            >
              {listening ? (
                <StopIcon sx={{ fontSize: 18 }} />
              ) : (
                <MicIcon sx={{ fontSize: 18 }} />
              )}
            </IconButton>
          </Tooltip>

          {/* Send button */}
          <Tooltip title={loading ? t("coach_thinking") : t("coach_send")}>
            <span>
              <IconButton
                onClick={handleSend}
                disabled={!inputValue.trim() || loading}
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: 2,
                  flexShrink: 0,
                  background:
                    inputValue.trim() && !loading
                      ? "linear-gradient(135deg, #2563EB, #7c3aed)"
                      : undefined,
                  color:
                    inputValue.trim() && !loading ? "white" : "text.disabled",
                  "&:hover": {
                    background:
                      inputValue.trim() && !loading
                        ? "linear-gradient(135deg, #1d4ed8, #6d28d9)"
                        : undefined,
                  },
                  transition: "all 0.2s",
                }}
              >
                {loading ? (
                  <CircularProgress size={17} color="inherit" />
                ) : (
                  <SendIcon sx={{ fontSize: 18 }} />
                )}
              </IconButton>
            </span>
          </Tooltip>
        </Paper>

        <Typography
          variant="caption"
          color="text.disabled"
          display="block"
          textAlign="center"
          mt={1.5}
        >
          {t("coach_footer_tip")}
        </Typography>
      </Box>
    </Box>
  );
}
