-- ═══════════════════════════════════════════════════════════════════════════════
--
--   ████████╗ █████╗ ██████╗
--   ╚══██╔══╝██╔══██╗██╔══██╗
--      ██║   ███████║██████╔╝
--      ██║   ██╔══██║██╔═══╝
--      ██║   ██║  ██║██║
--      ╚═╝   ╚═╝  ╚═╝╚═╝
--
--   TAP — Translate · Adapt · Precision
--   Supabase Database Schema v1.0
--
-- ═══════════════════════════════════════════════════════════════════════════════
--
--  SETUP INSTRUCTIONS:
--  1. Create a Supabase project at https://supabase.com
--  2. Go to SQL Editor (Dashboard → SQL Editor)
--  3. Paste this entire file and click "Run"
--  4. Copy your project URL and anon key from Dashboard → Settings → API
--  5. Add them to your .env file:
--       SUPABASE_URL=https://your-project.supabase.co
--       SUPABASE_ANON_KEY=your-anon-key
--
--  AUTH SETUP (recommended for development):
--  • Dashboard → Authentication → Providers → Email → enable
--  • Dashboard → Authentication → Settings → disable "Confirm email"
--    (allows instant signup without email verification during dev)
--
--  WHAT THIS CREATES:
--  • profiles         — user display name, avatar, bio
--  • user_api_keys    — per-user AI provider API keys (Gemini, Claude, etc.)
--  • chats            — per-user chat history with full HTML content
--  • Triggers         — auto-creates profile + keys row on signup
--  • RLS policies     — each user can only access their own data
--
-- ═══════════════════════════════════════════════════════════════════════════════


-- ─────────────────────────────────────────────────────────────────────────────
-- 0. EXTENSIONS
-- ─────────────────────────────────────────────────────────────────────────────
-- pgcrypto is enabled by default in Supabase, but ensure it's available
-- for gen_random_uuid() used by the chats table.

create extension if not exists "pgcrypto";


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. HELPER FUNCTION: auto-update `updated_at` timestamp
-- ─────────────────────────────────────────────────────────────────────────────
-- Shared by all tables. Automatically sets `updated_at = now()` on every UPDATE.

create or replace function public.update_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. TABLE: profiles
-- ─────────────────────────────────────────────────────────────────────────────
-- Stores public user info. One row per user, linked to auth.users.
-- Auto-created on signup via trigger (see section 5).
--
-- Columns:
--   id            — matches auth.users.id (UUID)
--   email         — synced from auth on signup
--   display_name  — user-editable name shown in UI
--   avatar_url    — optional profile picture URL
--   bio           — optional short bio
--   created_at    — when the profile was created
--   updated_at    — auto-updated on every change

create table if not exists public.profiles (
  id           uuid primary key references auth.users(id) on delete cascade,
  email        text,
  display_name text not null default '',
  avatar_url   text not null default '',
  bio          text not null default '',
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

comment on table public.profiles is 'User profiles — display name, avatar, bio. Auto-created on signup.';

-- Row Level Security: users can only read/write their own profile
alter table public.profiles enable row level security;

drop policy if exists "profiles_select" on public.profiles;
create policy "profiles_select" on public.profiles
  for select using (auth.uid() = id);

drop policy if exists "profiles_insert" on public.profiles;
create policy "profiles_insert" on public.profiles
  for insert with check (auth.uid() = id);

drop policy if exists "profiles_update" on public.profiles;
create policy "profiles_update" on public.profiles
  for update using (auth.uid() = id);

-- Auto-update timestamp trigger
drop trigger if exists profiles_updated_at on public.profiles;
create trigger profiles_updated_at
  before update on public.profiles
  for each row execute function public.update_updated_at();


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. TABLE: user_api_keys
-- ─────────────────────────────────────────────────────────────────────────────
-- Per-user AI provider API keys. Each user enters their own keys via the UI.
-- Keys are stored in Supabase (encrypted at rest), fetched by the frontend,
-- and sent per-request to the backend. The backend never stores them.
-- Auto-created on signup via trigger (see section 5).
--
-- Columns:
--   user_id     — matches auth.users.id (UUID)
--   gemini      — Google Gemini / AI Studio API key
--   claude      — Anthropic Claude API key
--   groq        — Groq API key
--   openrouter  — OpenRouter API key
--   updated_at  — auto-updated on every change

create table if not exists public.user_api_keys (
  user_id    uuid primary key references auth.users(id) on delete cascade,
  gemini     text not null default '',
  claude     text not null default '',
  groq       text not null default '',
  openrouter text not null default '',
  updated_at timestamptz not null default now()
);

comment on table public.user_api_keys is 'Per-user AI provider API keys. RLS ensures only the owner can access.';

-- Row Level Security: users can only access their own keys
alter table public.user_api_keys enable row level security;

drop policy if exists "keys_select" on public.user_api_keys;
create policy "keys_select" on public.user_api_keys
  for select using (auth.uid() = user_id);

drop policy if exists "keys_insert" on public.user_api_keys;
create policy "keys_insert" on public.user_api_keys
  for insert with check (auth.uid() = user_id);

drop policy if exists "keys_update" on public.user_api_keys;
create policy "keys_update" on public.user_api_keys
  for update using (auth.uid() = user_id);

-- Auto-update timestamp trigger
drop trigger if exists keys_updated_at on public.user_api_keys;
create trigger keys_updated_at
  before update on public.user_api_keys
  for each row execute function public.update_updated_at();


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. TABLE: chats
-- ─────────────────────────────────────────────────────────────────────────────
-- Per-user chat history. Each chat stores its full rendered HTML content
-- so switching between chats is instant (no re-rendering needed).
--
-- Columns:
--   id          — UUID primary key (generated by frontend via crypto.randomUUID())
--   user_id     — owner (references auth.users)
--   title       — chat title (auto-set from first prompt, user-editable)
--   chat_date   — human-readable date string (e.g. "Mar 22")
--   html        — full rendered chat HTML content
--   created_at  — when the chat was created
--   updated_at  — auto-updated on every change (used for sort order)

create table if not exists public.chats (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  title      text not null default 'New conversation',
  chat_date  text not null default '',
  html       text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

comment on table public.chats is 'Per-user chat history with rendered HTML content.';

-- Row Level Security: users can only access their own chats
alter table public.chats enable row level security;

drop policy if exists "chats_select" on public.chats;
create policy "chats_select" on public.chats
  for select using (auth.uid() = user_id);

drop policy if exists "chats_insert" on public.chats;
create policy "chats_insert" on public.chats
  for insert with check (auth.uid() = user_id);

drop policy if exists "chats_update" on public.chats;
create policy "chats_update" on public.chats
  for update using (auth.uid() = user_id);

drop policy if exists "chats_delete" on public.chats;
create policy "chats_delete" on public.chats
  for delete using (auth.uid() = user_id);

-- Index for fast lookups by user (used on every page load)
create index if not exists idx_chats_user_id on public.chats(user_id);

-- Auto-update timestamp trigger
drop trigger if exists chats_updated_at on public.chats;
create trigger chats_updated_at
  before update on public.chats
  for each row execute function public.update_updated_at();


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. TRIGGER: auto-create profile + api_keys on signup
-- ─────────────────────────────────────────────────────────────────────────────
-- When a new user signs up via Supabase Auth, this trigger fires and
-- automatically creates:
--   • A row in `profiles` with their email and display name
--   • A row in `user_api_keys` with empty keys (user fills them in later)
--
-- The display_name is pulled from signup metadata if provided,
-- otherwise defaults to the part before @ in their email.

create or replace function public.handle_new_user()
returns trigger as $$
begin
  -- Create profile
  insert into public.profiles (id, email, display_name)
  values (
    new.id,
    new.email,
    coalesce(
      nullif(trim(new.raw_user_meta_data->>'display_name'), ''),
      split_part(new.email, '@', 1)
    )
  );

  -- Create empty API keys row
  insert into public.user_api_keys (user_id)
  values (new.id);

  return new;
end;
$$ language plpgsql security definer;

comment on function public.handle_new_user() is 'Auto-creates profile and api_keys row when a new user signs up.';

-- Attach to auth.users
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. VERIFICATION
-- ─────────────────────────────────────────────────────────────────────────────
-- After running this script, verify in your Supabase dashboard:
--
-- Tables (Database → Tables):
--   ✓ profiles
--   ✓ user_api_keys
--   ✓ chats
--
-- Functions (Database → Functions):
--   ✓ handle_new_user()
--   ✓ update_updated_at()
--
-- Triggers:
--   ✓ on_auth_user_created  (on auth.users → creates profile + keys)
--   ✓ profiles_updated_at   (on profiles → auto-updates timestamp)
--   ✓ keys_updated_at       (on user_api_keys → auto-updates timestamp)
--   ✓ chats_updated_at      (on chats → auto-updates timestamp)
--
-- RLS Policies (all tables have row-level security enabled):
--   ✓ profiles:       select, insert, update (where auth.uid() = id)
--   ✓ user_api_keys:  select, insert, update (where auth.uid() = user_id)
--   ✓ chats:          select, insert, update, delete (where auth.uid() = user_id)
--
-- Quick test:
--   1. Sign up a user via your app
--   2. Check profiles table — should have a new row
--   3. Check user_api_keys table — should have a new row with empty keys
--   4. Add an API key via the UI, refresh user_api_keys — key should appear
--   5. Send a message, check chats table — chat should be saved
--   6. Log out and log back in — chat history should load


-- ═══════════════════════════════════════════════════════════════════════════════
-- END OF SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════
