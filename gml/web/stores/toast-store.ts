"use client";

import { create } from "zustand";

export type ToastKind = "success" | "error" | "info";

export interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  desc?: string;
}

interface ToastState {
  toasts: Toast[];
  push: (t: Omit<Toast, "id">) => void;
  dismiss: (id: number) => void;
}

let seq = 0;

export const useToasts = create<ToastState>((set) => ({
  toasts: [],
  push: (t) => {
    const id = ++seq;
    set((s) => ({ toasts: [...s.toasts, { ...t, id }] }));
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) }));
    }, 4500);
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })),
}));

/** Imperative helper for use outside React (mutation callbacks, etc.). */
export const toast = {
  success: (title: string, desc?: string) =>
    useToasts.getState().push({ kind: "success", title, desc }),
  error: (title: string, desc?: string) =>
    useToasts.getState().push({ kind: "error", title, desc }),
  info: (title: string, desc?: string) =>
    useToasts.getState().push({ kind: "info", title, desc }),
};
