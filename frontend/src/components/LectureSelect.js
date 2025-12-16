import React, { useEffect, useState } from "react";
import { TextField, MenuItem, CircularProgress } from "@mui/material";
import API from "../api/api";

export default function LectureSelect({ value, onChange, label = "" }) {
  const [lectures, setLectures] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      try {
        const res = await API.get("lectures/");
        if (!mounted) return;
        setLectures(res.data || []);
      } catch (e) {
        console.error("Failed to load lectures", e);
        setLectures([]);
      }
      setLoading(false);
    };
    load();
    return () => (mounted = false);
  }, []);

  return (
    <TextField
      select
      label={label}
      value={value || ""}
      onChange={(e) => onChange && onChange(e.target.value)}
      fullWidth
      SelectProps={{
        displayEmpty: true,
        renderValue: (selected) => {
          if (!selected)
            return <span style={{ color: "#9ca3af" }}>Lecture Note</span>;
          const lec = lectures.find((l) => String(l.id) === String(selected));
          return lec ? lec.title || `Lecture ${lec.id}` : selected;
        },
      }}
      InputProps={{
        endAdornment: loading ? <CircularProgress size={18} /> : null,
      }}
    >
      <MenuItem value="">Select a lecture</MenuItem>
      {lectures.map((lec) => (
        <MenuItem key={lec.id} value={lec.id}>
          {lec.title || `Lecture ${lec.id}`}
        </MenuItem>
      ))}
    </TextField>
  );
}
