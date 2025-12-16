// App.jsx
import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import SidebarLayout from "./layout/SidebarLayout";

// Page stubs - create these components or replace with your existing pages
import Dashboard from "./pages/Dashboard";
import UploadPage from "./pages/UploadNote";
import QuestionsPage from "./pages/GenerateQuestions";
import QuizPage from "./pages/Quiz";
import WeakTopicsPage from "./pages/WeakTopics";
import StudyPlanPage from "./pages/StudyPlan";
import QuizWrapper from "./pages/QuizWrapper";
import StudyPlanInteractive from "./pages/StudyPlanInteractive";
import QuizEntry from "./pages/QuizEntry";
import WeakTopicsEntry from "./pages/WeakTopicsEntry";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Profile from "./pages/Profile";
import GoogleCallback from "./pages/GoogleCallback";
import GoogleLogin from "./pages/GoogleLogin";
import QuizResult from "./pages/QuizResult";
import Lectures from "./pages/Lectures";
import Flashcards from "./pages/Flashcards";
import SummarizeLectures from "./pages/SummarizeLectures";
import ExamPreparation from "./pages/ExamPreparation";
import LandingPage from "./pages/LandingPage";

import { ThemeProvider, CssBaseline } from "@mui/material";
import theme from "./theme";

export default function App() {
  return (
    <AuthProvider>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          <Routes>
            {/* Public routes */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/google-login" element={<GoogleLogin />} />
            <Route path="/auth/google/callback" element={<GoogleCallback />} />
            
            {/* Protected routes with sidebar */}
            <Route path="/" element={<SidebarLayout />}>
              <Route path="upload" element={<UploadPage />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="questions" element={<QuestionsPage />} />
              <Route path="quiz/:noteId" element={<QuizWrapper />} />
              {/* Quiz page moved out to prevent navigation via sidebar */}
              <Route path="quiz-entry" element={<QuizEntry />} />
              <Route path="weak-topics/:noteId" element={<WeakTopicsPage />} />
              <Route path="study-plan" element={<StudyPlanInteractive />} />
              <Route path="weak-topics" element={<WeakTopicsEntry />} />
              <Route path="profile" element={<Profile />} />
              <Route path="quiz-result" element={<QuizResult />} />
              <Route path="lectures" element={<Lectures />} />
              <Route path="flashcards" element={<Flashcards />} />
              <Route path="summarize" element={<SummarizeLectures />} />
              <Route path="exam-preparation" element={<ExamPreparation />} />
            </Route>
            
            {/* Standalone Quiz Route (No Sidebar) */}
            <Route path="/quiz" element={<QuizPage />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </AuthProvider>
  );
}
