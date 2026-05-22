import React from "react";
import { Tabs, Tab, Box } from "@mui/material";

export default function SectionTabs({ value, onChange, tabs = [] }) {
  return (
    <Box sx={{ mb: 2 }}>
      <Tabs
        value={value}
        onChange={onChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          minHeight: 42,
          "& .MuiTab-root": {
            minHeight: 42,
            textTransform: "none",
            fontWeight: 700,
            borderRadius: 2,
            mr: 1,
          },
        }}
      >
        {tabs.map((tab) => (
          <Tab key={tab.value} label={tab.label} value={tab.value} />
        ))}
      </Tabs>
    </Box>
  );
}
