import React from "react";
import { Box } from "@mui/material";
import EmptyState from "./EmptyState";
import ErrorState from "./ErrorState";
import LoadingSkeletonPack from "./LoadingSkeletonPack";

export default function AsyncContent({
  loading,
  error,
  isEmpty,
  onRetry,
  loadingRows = 3,
  skeletonHeight = 120,
  emptyTitle,
  emptyMessage,
  emptyActionLabel,
  onEmptyAction,
  children,
}) {
  if (loading) {
    return (
      <LoadingSkeletonPack rows={loadingRows} cardHeight={skeletonHeight} />
    );
  }

  if (error) {
    return <ErrorState message={error} onRetry={onRetry} />;
  }

  if (isEmpty) {
    return (
      <EmptyState
        title={emptyTitle}
        message={emptyMessage}
        actionLabel={emptyActionLabel}
        onAction={onEmptyAction}
      />
    );
  }

  return <Box>{children}</Box>;
}
