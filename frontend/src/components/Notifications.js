import React, { useState, useEffect, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import {
  Box,
  IconButton,
  Badge,
  Menu,
  MenuItem,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Divider,
  Button,
  Tooltip,
  CircularProgress,
} from "@mui/material";
import NotificationsIcon from "@mui/icons-material/Notifications";
import DoneAllIcon from "@mui/icons-material/DoneAll";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import CircleIcon from "@mui/icons-material/Circle";
import NotificationsNoneIcon from "@mui/icons-material/NotificationsNone";

export default function Notifications() {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [anchorEl, setAnchorEl] = useState(null);
  const [loading, setLoading] = useState(false);
  const { isAuthenticated, api } = useAuth();
  const open = Boolean(anchorEl);

  // Poll for notifications every 60 seconds
  useEffect(() => {
    if (isAuthenticated && api) {
      fetchNotifications();
      const interval = setInterval(fetchNotifications, 60000);
      
      // Listen for custom refresh event
      const handleRefresh = () => {
        // console.log('Refreshing notifications...');
        fetchNotifications();
      };
      window.addEventListener('refreshNotifications', handleRefresh);
      
      return () => {
        clearInterval(interval);
        window.removeEventListener('refreshNotifications', handleRefresh);
      };
    }
  }, [isAuthenticated, api]);

  const fetchNotifications = async () => {
    if (!api) return;
    
    try {
      const response = await api.get("notifications/");
      setNotifications(response.data);
      setUnreadCount(response.data.filter((n) => !n.is_read).length);
    } catch (error) {
      console.error("Failed to fetch notifications:", error);
      // Don't show error to user if it's just an auth issue
      if (error.response?.status !== 401) {
        console.error("Unexpected error fetching notifications:", error);
      }
    }
  };

  const handleOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const markAsRead = async (id, event) => {
    event.stopPropagation();
    if (!api) return;
    
    try {
      await api.post(`http://localhost:8000/api/notifications/${id}/mark-read/`);
      // Optimistic update
      setNotifications((prev) =>
        prev.map((n) => (n.id === id ? { ...n, is_read: true } : n))
      );
      setUnreadCount((prev) => Math.max(0, prev - 1));
    } catch (error) {
      console.error("Failed to mark as read:", error);
    }
  };

  const markAllAsRead = async () => {
    if (!api) return;
    
    try {
      await api.post("http://localhost:8000/api/notifications/mark-all-read/");
      setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
      setUnreadCount(0);
    } catch (error) {
      console.error("Failed to mark all as read:", error);
    }
  };

  const deleteNotification = async (id, event) => {
    event.stopPropagation();
    if (!api) return;
    
    try {
      await api.delete(`http://localhost:8000/api/notifications/${id}/delete/`);
      setNotifications((prev) => prev.filter((n) => n.id !== id));
      // Recalculate unread count if we deleted an unread one
      const deletedNote = notifications.find((n) => n.id === id);
      if (deletedNote && !deletedNote.is_read) {
        setUnreadCount((prev) => Math.max(0, prev - 1));
      }
    } catch (error) {
      console.error("Failed to delete notification:", error);
    }
  };

  if (!isAuthenticated) return null;

  return (
    <>
      <Tooltip title="Notifications">
        <IconButton
          onClick={handleOpen}
          sx={{
            color: "white",
            "&:hover": { background: "rgba(255,255,255,0.1)" },
          }}
        >
          <Badge badgeContent={unreadCount} color="error">
            <NotificationsIcon />
          </Badge>
        </IconButton>
      </Tooltip>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        PaperProps={{
          elevation: 0,
          sx: {
            width: 360,
            maxHeight: 500,
            overflow: "visible",
            filter: "drop-shadow(0px 2px 8px rgba(0,0,0,0.32))",
            mt: 1.5,
            borderRadius: 3,
            "&:before": {
              content: '""',
              display: "block",
              position: "absolute",
              top: 0,
              right: 14,
              width: 10,
              height: 10,
              bgcolor: "background.paper",
              transform: "translateY(-50%) rotate(45deg)",
              zIndex: 0,
            },
          },
        }}
        transformOrigin={{ horizontal: "right", vertical: "top" }}
        anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
      >
        <Box
          sx={{
            p: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: "1px solid #eee",
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: "bold", fontSize: "1rem" }}>
            Notifications
          </Typography>
          {unreadCount > 0 && (
            <Button
              size="small"
              startIcon={<DoneAllIcon />}
              onClick={markAllAsRead}
              sx={{ textTransform: "none", fontSize: "0.8rem" }}
            >
              Mark all read
            </Button>
          )}
        </Box>

        <List sx={{ p: 0, maxHeight: 400, overflowY: "auto" }}>
          {notifications.length === 0 ? (
            <Box
              sx={{
                p: 4,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                color: "text.secondary",
              }}
            >
              <NotificationsNoneIcon sx={{ fontSize: 48, mb: 1, opacity: 0.5 }} />
              <Typography variant="body2">No notifications yet</Typography>
            </Box>
          ) : (
            notifications.map((notification) => (
              <React.Fragment key={notification.id}>
                <ListItem
                  alignItems="flex-start"
                  sx={{
                    bgcolor: notification.is_read ? "transparent" : "action.hover",
                    transition: "background-color 0.2s",
                    "&:hover": {
                      bgcolor: "action.selected",
                      "& .delete-btn": { opacity: 1 },
                    },
                    cursor: "default",
                  }}
                  secondaryAction={
                    <IconButton
                      edge="end"
                      size="small"
                      onClick={(e) => deleteNotification(notification.id, e)}
                      className="delete-btn"
                      sx={{ opacity: 0, transition: "opacity 0.2s" }}
                    >
                      <DeleteOutlineIcon fontSize="small" />
                    </IconButton>
                  }
                >
                  <ListItemAvatar sx={{ minWidth: 40 }}>
                    {!notification.is_read ? (
                      <CircleIcon
                        sx={{ fontSize: 12, color: "primary.main", mt: 1 }}
                      />
                    ) : (
                      <Box sx={{ width: 12 }} />
                    )}
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: notification.is_read ? 400 : 600,
                          color: "text.primary",
                        }}
                      >
                        {notification.message}
                      </Typography>
                    }
                    secondary={
                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          mt: 0.5,
                        }}
                      >
                        <Typography variant="caption" color="text.secondary">
                          {new Date(notification.created_at).toLocaleDateString()}{" "}
                          {new Date(notification.created_at).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </Typography>
                        {!notification.is_read && (
                          <Button
                            size="small"
                            onClick={(e) => markAsRead(notification.id, e)}
                            sx={{
                              fontSize: "0.7rem",
                              minWidth: "auto",
                              p: 0.5,
                              height: "auto",
                            }}
                          >
                            Mark read
                          </Button>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
                <Divider component="li" />
              </React.Fragment>
            ))
          )}
        </List>
      </Menu>
    </>
  );
}
