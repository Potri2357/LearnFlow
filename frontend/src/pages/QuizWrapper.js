// src/pages/QuizWrapper.js
import React, { useEffect } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import Quiz from "./Quiz";

export default function QuizWrapper() {
  const { noteId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Redirect old path-based route to query parameter route
    if (noteId) {
      const searchParams = new URLSearchParams(location.search);
      const n = searchParams.get("n") || 10;
      navigate(`/quiz?noteId=${noteId}&n=${n}`, { replace: true });
    }
  }, [noteId, navigate, location.search]);

  // If noteId is in path, show loading while redirecting
  if (noteId) {
    return <div>Redirecting...</div>;
  }

  // Otherwise render Quiz normally (it will get noteId from query params)
  return <Quiz />;
}
