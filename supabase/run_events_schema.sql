create table if not exists public.runs (
  run_id text primary key,
  graph_id text not null,
  agent_id text,
  agent_name text,
  parent_run_id text,
  status text not null,
  status_reason text,
  started_at text,
  ended_at text,
  runtime_instance_id text,
  last_heartbeat_at text,
  input_payload jsonb,
  final_output jsonb,
  terminal_error jsonb,
  current_node_id text,
  current_edge_id text,
  state_snapshot jsonb,
  created_at text not null
);

create index if not exists runs_graph_id_created_at_idx on public.runs (graph_id, created_at desc);
create index if not exists runs_parent_run_id_idx on public.runs (parent_run_id);

create table if not exists public.run_events (
  id bigint generated always as identity primary key,
  run_id text not null references public.runs(run_id) on delete cascade,
  sequence_number bigint not null,
  event_type text not null,
  timestamp text not null,
  agent_id text,
  parent_run_id text,
  summary text not null,
  payload jsonb not null default '{}'::jsonb,
  unique (run_id, sequence_number)
);

create index if not exists run_events_run_id_sequence_idx on public.run_events (run_id, sequence_number asc);
create index if not exists run_events_parent_run_id_idx on public.run_events (parent_run_id);
