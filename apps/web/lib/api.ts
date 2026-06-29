export type ApiEnvelope<T> = {
  status: string;
  data: T;
};

const API_BASE = typeof window !== 'undefined' 
  ? `http://${window.location.hostname}:8000` 
  : (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000");

let cachedToken: string | null = null;

export async function fetchToken(): Promise<string> {
  if (cachedToken) return cachedToken;
  const response = await fetch(`${API_BASE}/api/v1/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: "admin", password: "warehouse" }),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Unable to acquire API token");
  }
  const payload = (await response.json()) as { access_token: string };
  cachedToken = payload.access_token;
  return cachedToken;
}

export async function callApi<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`API call failed: ${path}`);
  }
  const payload = (await response.json()) as ApiEnvelope<T>;
  return payload.data;
}

export function getWebSocketUrl(): string {
  return API_BASE.replace("http", "ws") + "/ws/events";
}

