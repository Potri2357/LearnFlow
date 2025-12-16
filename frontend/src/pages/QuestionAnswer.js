import React, { useState } from "react";
import API from "../api/api";
import {
  Container,
  TextField,
  Button,
  Card,
  CardContent,
  Typography,
  Snackbar,
  CircularProgress,
} from "@mui/material";

function QuestionAnswer() {
  const [questionId, setQuestionId] = useState("");
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      const res = await API.post("submit-answer/", {
        question_id: questionId,
        user_answer: answer,
      });
      setResult(res.data.correct);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" style={{ marginTop: 40 }}>
      <Card>
        <CardContent>
          <Typography variant="h4">Submit Answer</Typography>

          <TextField
            fullWidth
            label="Question ID"
            margin="normal"
            onChange={(e) => setQuestionId(e.target.value)}
          />

          <TextField
            fullWidth
            label="Your Answer"
            margin="normal"
            multiline
            rows={4}
            onChange={(e) => setAnswer(e.target.value)}
          />

          <Button
            variant="contained"
            color="success"
            onClick={submit}
            disabled={loading}
            sx={{
              background: "linear-gradient(135deg, #16a34a, #4ade80)",
              borderRadius: "10px",
              py: 1.2,
              px: 4,
              fontWeight: "bold",
            }}
          >
            {loading ? (
              <CircularProgress size={18} color="inherit" />
            ) : (
              "Submit"
            )}
          </Button>

          {result !== null && (
            <Typography
              variant="h6"
              style={{ marginTop: 15, color: result ? "green" : "red" }}
            >
              {result ? "Correct Answer!" : "Wrong Answer!"}
            </Typography>
          )}
        </CardContent>
      </Card>
    </Container>
  );
}

export default QuestionAnswer;
