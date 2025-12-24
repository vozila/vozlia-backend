// src/lib/vozliaApi.ts
export type MeSettings = {
  agent_greeting: string;
  gmail_summary_enabled: boolean;
  gmail_account_id: string | null;
  realtime_prompt_addendum: string | null;
};

export type EmailAccount = {
  id: string;
  provider_type: string;          // "gmail", etc.
  oauth_provider: string | null;  // "google"
  email_address: string;
  is_primary: boolean;
  is_active: boolean;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

function assertApiBase() {
  if (!API_BASE) {
    throw new Error(
      "Missing NEXT_PUBLIC_API_BASE_URL. Set it in your environment (e.g. https://api.vozlia.com)."
    );
  }
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  assertApiBase();

  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const msg = text || res.statusText || `HTTP ${res.status}`;
    const err = new Error(msg) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }

  // Some endpoints might return empty; but ours return JSON.
  return (await res.json()) as T;
}

export const VozliaAPI = {
  getMeSettings: () => api<MeSettings>("/me/settings"),

  updateGreeting: (text: string) =>
    api<MeSettings>("/me/settings/greeting", {
      method: "PUT",
      body: JSON.stringify({ text }),
    }),

  updateRealtimePrompt: (text: string) =>
    api<MeSettings>("/me/settings/realtime-prompt", {
      method: "PUT",
      body: JSON.stringify({ text }),
    }),

  updateGmailSummaryEnabled: (enabled: boolean) =>
    api<MeSettings>("/me/settings/gmail-summary/enabled", {
      method: "PUT",
      body: JSON.stringify({ enabled }),
    }),

  listEmailAccounts: () => api<EmailAccount[]>("/me/email-accounts"),

  selectGmailAccount: (account_id: string) =>
    api<MeSettings>("/me/settings/gmail-account", {
      method: "PUT",
      body: JSON.stringify({ account_id }),
    }),
};
