// Routes: POST /api/presence (heartbeat in, online count out) and GET /api/presence/history (visits per country).
import { call } from "../shared/http";
import type { PresenceHistoryOut, PresenceOut } from "../generated/wire";

export const heartbeat = (clientId: string) =>
  call<PresenceOut>("/api/presence", { method: "POST", body: JSON.stringify({ client_id: clientId }) });

export const visitHistory = () => call<PresenceHistoryOut>("/api/presence/history");
