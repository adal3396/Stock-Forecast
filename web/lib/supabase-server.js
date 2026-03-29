import { createClient } from "@supabase/supabase-js";

function getConfig() {
  const url =
    process.env.NEXT_PUBLIC_SUPABASE_URL?.trim() ||
    process.env.SUPABASE_URL?.trim();
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY?.trim();
  if (!url) {
    throw new Error(
      "Set NEXT_PUBLIC_SUPABASE_URL (or SUPABASE_URL) in environment variables."
    );
  }
  if (!key) {
    throw new Error(
      "Set SUPABASE_SERVICE_ROLE_KEY (Supabase → Project Settings → API → service_role). Never expose this key to the browser."
    );
  }
  return { url, key };
}

/** Server-only Supabase client (bypasses RLS). Use only in API routes / server code. */
export function createServiceClient() {
  const { url, key } = getConfig();
  return createClient(url, key, {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
    },
  });
}
