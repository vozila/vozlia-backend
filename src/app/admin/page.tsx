// src/app/admin/page.tsx
"use client";

import React, { useEffect, useMemo, useState } from "react";
import { EmailAccount, MeSettings, VozliaAPI } from "@/lib/vozliaApi";

type Toast = { type: "success" | "error"; message: string };

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

/**
 * Redirect user to backend OAuth login.
 * We *try* to pass a `next=` return URL; if your backend ignores it, it will still work (just returns wherever backend defaults).
 */
function buildLoginUrl() {
  const next = encodeURIComponent(`${window.location.origin}/admin`);
  if (!API_BASE) return `/admin/login?next=${next}`; // fallback (won't work cross-domain, but avoids crashing)
  return `${API_BASE}/admin/login?next=${next}`;
}

function ToastView({ toast, onClose }: { toast: Toast | null; onClose: () => void }) {
  if (!toast) return null;
  return (
    <div
      className={`fixed right-4 top-4 z-50 w-[360px] rounded-xl border p-3 shadow-lg ${
        toast.type === "success" ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"
      }`}
      role="status"
      aria-live="polite"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="text-sm">
          <div className="font-semibold">{toast.type === "success" ? "Saved" : "Error"}</div>
          <div className="mt-0.5 opacity-90">{toast.message}</div>
        </div>
        <button
          className="rounded-md px-2 py-1 text-xs hover:bg-black/5"
          onClick={onClose}
          aria-label="Dismiss"
        >
          ✕
        </button>
      </div>
    </div>
  );
}

function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-1">
        <div className="text-lg font-semibold">{title}</div>
        {description ? <div className="text-sm text-gray-600">{description}</div> : null}
      </div>
      <div className="mt-4">{children}</div>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="mb-2 text-sm font-medium text-gray-800">{children}</div>;
}

export default function AdminSettingsPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);

  const [toast, setToast] = useState<Toast | null>(null);

  const [settings, setSettings] = useState<MeSettings | null>(null);
  const [emailAccounts, setEmailAccounts] = useState<EmailAccount[]>([]);

  // ✅ IMPORTANT: this was missing in your current file and causes runtime failure
  const [unauthenticated, setUnauthenticated] = useState(false);

  // local edit state
  const [greetingDraft, setGreetingDraft] = useState("");
  const [realtimeDraft, setRealtimeDraft] = useState("");
  const [gmailEnabledDraft, setGmailEnabledDraft] = useState(false);
  const [gmailAccountDraft, setGmailAccountDraft] = useState<string>("");

  const activeGmailAccounts = useMemo(() => {
    return emailAccounts
      .filter((a) => a.is_active)
      .filter(
        (a) =>
          (a.provider_type || "").toLowerCase() === "gmail" ||
          (a.oauth_provider || "").toLowerCase() === "google"
      )
      .sort((a, b) => a.email_address.localeCompare(b.email_address));
  }, [emailAccounts]);

  function showToast(t: Toast) {
    setToast(t);
    window.setTimeout(() => setToast(null), 3000);
  }

  async function loadAll() {
    setLoading(true);
    setUnauthenticated(false);

    try {
      const [s, accts] = await Promise.all([VozliaAPI.getMeSettings(), VozliaAPI.listEmailAccounts()]);
      setSettings(s);
      setEmailAccounts(accts);

      setGreetingDraft(s.agent_greeting || "");
      setRealtimeDraft(s.realtime_prompt_addendum || "");
      setGmailEnabledDraft(!!s.gmail_summary_enabled);
      setGmailAccountDraft(s.gmail_account_id || "");
    } catch (e: any) {
      const status = e?.status;

      if (status === 401) {
        setUnauthenticated(true);
        showToast({
          type: "error",
          message: "Not authenticated. Please log in with Google.",
        });
      } else {
        showToast({ type: "error", message: e?.message || "Failed to load settings." });
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const greetingDirty = settings ? greetingDraft !== (settings.agent_greeting || "") : false;
  const realtimeDirty = settings ? realtimeDraft !== (settings.realtime_prompt_addendum || "") : false;
  const gmailEnabledDirty = settings ? gmailEnabledDraft !== !!settings.gmail_summary_enabled : false;
  const gmailAccountDirty = settings ? gmailAccountDraft !== (settings.gmail_account_id || "") : false;

  async function saveGreeting() {
    setSaving("greeting");
    try {
      const s = await VozliaAPI.updateGreeting(greetingDraft);
      setSettings(s);
      setGreetingDraft(s.agent_greeting || "");
      showToast({ type: "success", message: "Greeting updated." });
    } catch (e: any) {
      showToast({ type: "error", message: e?.message || "Failed to update greeting." });
    } finally {
      setSaving(null);
    }
  }

  async function saveRealtimePrompt() {
    setSaving("realtime");
    try {
      const s = await VozliaAPI.updateRealtimePrompt(realtimeDraft);
      setSettings(s);
      setRealtimeDraft(s.realtime_prompt_addendum || "");
      showToast({ type: "success", message: "Opening rule updated." });
    } catch (e: any) {
      showToast({ type: "error", message: e?.message || "Failed to update opening rule." });
    } finally {
      setSaving(null);
    }
  }

  async function saveGmailEnabled() {
    setSaving("gmailEnabled");
    try {
      const s = await VozliaAPI.updateGmailSummaryEnabled(gmailEnabledDraft);
      setSettings(s);
      setGmailEnabledDraft(!!s.gmail_summary_enabled);
      showToast({ type: "success", message: "Email summaries setting updated." });
    } catch (e: any) {
      showToast({ type: "error", message: e?.message || "Failed to update email summaries toggle." });
    } finally {
      setSaving(null);
    }
  }

  async function saveGmailAccount() {
    setSaving("gmailAccount");
    try {
      if (!gmailAccountDraft) {
        showToast({ type: "error", message: "Select an inbox first." });
        return;
      }
      const s = await VozliaAPI.selectGmailAccount(gmailAccountDraft);
      setSettings(s);
      setGmailAccountDraft(s.gmail_account_id || "");
      showToast({ type: "success", message: "Gmail inbox selected." });
    } catch (e: any) {
      showToast({ type: "error", message: e?.message || "Failed to select Gmail inbox." });
    } finally {
      setSaving(null);
    }
  }

  const showControls = !loading && !unauthenticated;

  return (
    <div className="min-h-screen bg-gray-50">
      <ToastView toast={toast} onClose={() => setToast(null)} />

      <div className="mx-auto max-w-5xl px-4 py-10">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-2xl font-bold">Admin</div>
            <div className="mt-1 text-sm text-gray-600">Vozlia assistant settings</div>
          </div>

          <button
            className="rounded-xl border bg-white px-4 py-2 text-sm shadow-sm hover:bg-gray-50 disabled:opacity-50"
            onClick={loadAll}
            disabled={loading || !!saving}
          >
            Refresh
          </button>
        </div>

        {loading ? (
          <div className="mt-8 rounded-2xl border bg-white p-6 shadow-sm">
            <div className="text-sm text-gray-600">Loading settings…</div>
          </div>
        ) : unauthenticated ? (
          <div className="mt-8 rounded-2xl border bg-white p-6 shadow-sm">
            <div className="text-lg font-semibold">Login required</div>
            <div className="mt-2 text-sm text-gray-600">
              You’re not authenticated for the API. Click below to log in with Google.
            </div>

            <div className="mt-4 flex flex-col gap-2">
              <button
                className="rounded-xl bg-black px-4 py-2 text-sm text-white"
                onClick={() => {
                  if (!API_BASE) {
                    showToast({
                      type: "error",
                      message:
                        "Missing NEXT_PUBLIC_API_BASE_URL on vozlia-admin. Set it to your backend URL (Render or api.vozlia.com).",
                    });
                    return;
                  }
                  window.location.href = buildLoginUrl();
                }}
              >
                Login with Google
              </button>

              <div className="text-xs text-gray-500">
                API base: <code>{API_BASE || "(not set)"}</code>
              </div>
            </div>
          </div>
        ) : showControls ? (
          <div className="mt-8 grid grid-cols-1 gap-6">
            <Section title="Agent greeting" description="This is the default greeting the assistant uses (DB-backed).">
              <Label>Greeting</Label>
              <textarea
                className="h-28 w-full rounded-xl border p-3 text-sm outline-none focus:ring-2 focus:ring-black/10"
                value={greetingDraft}
                onChange={(e) => setGreetingDraft(e.target.value)}
                placeholder="Hi! Thanks for calling…"
              />
              <div className="mt-3 flex items-center justify-between gap-3">
                <div className="text-xs text-gray-500">{greetingDirty ? "Unsaved changes" : "Saved"}</div>
                <button
                  className="rounded-xl bg-black px-4 py-2 text-sm text-white disabled:opacity-50"
                  onClick={saveGreeting}
                  disabled={!greetingDirty || saving === "greeting"}
                >
                  {saving === "greeting" ? "Saving…" : "Save"}
                </button>
              </div>
            </Section>

            <Section
              title="Realtime opening rule"
              description="A short rule applied once at call start to steer the opening (DB-backed addendum)."
            >
              <Label>Opening rule (addendum)</Label>
              <textarea
                className="h-36 w-full rounded-xl border p-3 text-sm outline-none focus:ring-2 focus:ring-black/10"
                value={realtimeDraft}
                onChange={(e) => setRealtimeDraft(e.target.value)}
                placeholder='Example: "Open by saying: Thanks for calling {Business}. How can I help today?"'
              />
              <div className="mt-3 flex items-center justify-between gap-3">
                <div className="text-xs text-gray-500">{realtimeDirty ? "Unsaved changes" : "Saved"}</div>
                <button
                  className="rounded-xl bg-black px-4 py-2 text-sm text-white disabled:opacity-50"
                  onClick={saveRealtimePrompt}
                  disabled={!realtimeDirty || saving === "realtime"}
                >
                  {saving === "realtime" ? "Saving…" : "Save"}
                </button>
              </div>
            </Section>

            <Section title="Email summaries" description="Controls whether the assistant can summarize emails (DB-backed).">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-sm font-medium text-gray-900">Enable email summaries</div>
                  <div className="mt-1 text-xs text-gray-600">
                    If disabled, the assistant will not provide Gmail summaries.
                  </div>
                </div>

                <label className="inline-flex cursor-pointer items-center gap-2">
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={gmailEnabledDraft}
                    onChange={(e) => setGmailEnabledDraft(e.target.checked)}
                  />
                  <span className="text-sm">{gmailEnabledDraft ? "On" : "Off"}</span>
                </label>
              </div>

              <div className="mt-3 flex items-center justify-between gap-3">
                <div className="text-xs text-gray-500">{gmailEnabledDirty ? "Unsaved changes" : "Saved"}</div>
                <button
                  className="rounded-xl bg-black px-4 py-2 text-sm text-white disabled:opacity-50"
                  onClick={saveGmailEnabled}
                  disabled={!gmailEnabledDirty || saving === "gmailEnabled"}
                >
                  {saving === "gmailEnabled" ? "Saving…" : "Save"}
                </button>
              </div>
            </Section>

            <Section
              title="Gmail inbox selection"
              description="Select which connected Gmail inbox is used for summaries."
            >
              <Label>Inbox</Label>

              {activeGmailAccounts.length === 0 ? (
                <div className="rounded-xl border bg-gray-50 p-3 text-sm text-gray-700">
                  No active Gmail accounts found. Connect an account first.
                </div>
              ) : (
                <select
                  className="w-full rounded-xl border p-3 text-sm outline-none focus:ring-2 focus:ring-black/10"
                  value={gmailAccountDraft}
                  onChange={(e) => setGmailAccountDraft(e.target.value)}
                >
                  <option value="">Select an inbox…</option>
                  {activeGmailAccounts.map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.email_address}
                      {a.is_primary ? " (primary)" : ""}
                    </option>
                  ))}
                </select>
              )}

              <div className="mt-3 flex items-center justify-between gap-3">
                <div className="text-xs text-gray-500">{gmailAccountDirty ? "Unsaved changes" : "Saved"}</div>
                <button
                  className="rounded-xl bg-black px-4 py-2 text-sm text-white disabled:opacity-50"
                  onClick={saveGmailAccount}
                  disabled={!gmailAccountDirty || saving === "gmailAccount" || activeGmailAccounts.length === 0}
                >
                  {saving === "gmailAccount" ? "Saving…" : "Save"}
                </button>
              </div>
            </Section>

            <div className="rounded-2xl border bg-white p-5 text-sm text-gray-600 shadow-sm">
              <div className="font-medium text-gray-900">Notes</div>
              <ul className="mt-2 list-disc pl-5">
                <li>
                  All requests use cookies via <code>credentials: "include"</code>.
                </li>
                <li>
                  If you see 401s after logging in, confirm backend cookies + CORS allow credentials for{" "}
                  <code>admin.vozlia.com</code>.
                </li>
                <li>
                  Login button redirects to: <code>{API_BASE || "(missing API base)"}/admin/login</code>
                </li>
              </ul>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
