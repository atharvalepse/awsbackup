/**
 * Client-side auth token storage. The JWT from /auth/login|signup is kept in
 * localStorage and attached as `Authorization: Bearer` by lib/api.ts.
 */
const TOKEN_KEY = "gml.token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  if (typeof window !== "undefined") window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  if (typeof window !== "undefined") window.localStorage.removeItem(TOKEN_KEY);
}

export function isAuthed(): boolean {
  return getToken() !== null;
}
