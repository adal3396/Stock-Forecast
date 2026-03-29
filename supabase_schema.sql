-- Run this once in Supabase → SQL Editor → New query → Run.
-- Stores dashboard JSON blobs (same shape as the old MongoDB documents, without top-level `type` inside payload).

create table if not exists public.dashboard_documents (
  doc_type text primary key,
  payload jsonb not null,
  updated_at timestamptz not null default now()
);

create index if not exists dashboard_documents_updated_ix
  on public.dashboard_documents (updated_at desc);

alter table public.dashboard_documents enable row level security;

-- Anon/authenticated users: no direct access (dashboard is read via Next.js API + service role only).
-- The service_role key bypasses RLS.
